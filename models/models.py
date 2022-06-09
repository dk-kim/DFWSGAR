import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np

from .backbone import build_backbone
from .token_encoder import build_token_encoder


class DFGAR(nn.Module):
    def __init__(self, args):
        super(DFGAR, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim
        self.num_tokens = args.num_tokens

        # feature extraction
        self.backbone = build_backbone(args)
        self.token_encoder = build_token_encoder(args)
        self.query_embed = nn.Embedding(self.num_tokens, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.token_encoder.d_model, kernel_size=1)

        if self.dataset == 'volleyball':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        elif self.dataset == 'nba':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
        else:
            assert False

        self.self_attn = nn.MultiheadAttention(self.token_encoder.d_model, args.nheads_agg, dropout=args.drop_rate)
        self.dropout1 = nn.Dropout(args.drop_rate)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_class)

        self.relu = F.relu
        self.gelu = F.gelu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        src = self.input_proj(src)
        representations, _ = self.token_encoder(src, None, self.query_embed.weight, pos)
        # [1, B x T, K, F'], [1, B x T, K, H' x W']

        representations = representations.reshape(b, t, self.num_tokens, -1)                    # [B, T, K, D]

        if self.dataset == 'volleyball':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)                                # [B, D]
        elif self.dataset == 'nba':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = self.conv3(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)                                # [B x K, D]
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)                                # [B, D]

        representations = representations.reshape(b, -1)                                        # [B, K' x F]
        activities_scores = self.classifier(representations)                                    # [B, C]

        return activities_scores


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim

        # feature extraction
        self.backbone = build_backbone(args)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.backbone.num_channels, self.num_class)

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        representations = self.avg_pool(src)
        representations = representations.reshape(b, t, c)

        representations = representations.reshape(b * t, self.backbone.num_channels)        # [B, T, F]
        activities_scores = self.classifier(representations)
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)

        return activities_scores
