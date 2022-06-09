# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from spatial_correlation_sampler import SpatialCorrelationSampler

from .position_encoding import build_position_encoding


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation], pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if self.motion:
            self.layer_channel = [64, 128, 256, 512]

            self.channel_dim = self.layer_channel[self.motion_layer - 1]

            self.corr_input_proj = nn.Sequential(
                nn.Conv2d(self.channel_dim, self.corr_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

            self.neighbor_size = args.neighbor_size
            self.ps = 2 * args.neighbor_size + 1

            self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                                 stride=1, padding=0, dilation_patch=1)

            self.corr_output_proj = nn.Sequential(
                nn.Conv2d(self.ps * self.ps, self.channel_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

    def get_local_corr(self, x):
        x = self.corr_input_proj(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        corr = self.corr_output_proj(corr)

        return corr

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.motion:
            if self.motion_layer == 1:
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 2:
                x = self.layer2(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 3:
                x = self.layer2(x)
                x = self.layer3(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer4(x)
            elif self.motion_layer == 4:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x)
                x = x + corr
            else:
                assert False
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x


class MultiCorrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args):
        super(MultiCorrBackbone, self).__init__()

        backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation],
            pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer_channel = [64, 128, 256, 512]

        self.channel_dim = self.layer_channel[self.motion_layer - 1]

        self.corr_input_proj1 = nn.Sequential(
            nn.Conv2d(self.layer_channel[2], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_input_proj2 = nn.Sequential(
            nn.Conv2d(self.layer_channel[3], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.neighbor_size = args.neighbor_size
        self.ps = 2 * args.neighbor_size + 1

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                             stride=1, padding=0, dilation_patch=1)

        self.corr_output_proj1 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[2], kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_output_proj2 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[3], kernel_size=1, bias=False),
            nn.ReLU()
        )

    def get_local_corr(self, x, idx):
        if idx == 0:
            x = self.corr_input_proj1(x)
        else:
            x = self.corr_input_proj2(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        if idx == 0:
            corr = self.corr_output_proj1(corr)
        else:
            corr = self.corr_output_proj2(corr)

        return corr

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        corr = self.get_local_corr(x, 0)
        x = x + corr

        x = self.layer4(x)
        corr = self.get_local_corr(x, 1)
        x = x + corr

        return x


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        features = self[0](x)
        pos = self[1](features).to(x.dtype)

        return features, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if args.multi_corr:
        backbone = MultiCorrBackbone(args)
    else:
        backbone = Backbone(args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
