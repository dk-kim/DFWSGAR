import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']


def read_ids(path):
    file = open(path)
    values = file.readline()
    values = values.split(',')[:-1]
    values = list(map(int, values))

    return values


def nba_read_annotations(path, seqs):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split('\t')
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels


def nba_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames


class NBADataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(NBADataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        return len(self.frames)

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        vid, sid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(72), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
        else:
            if self.num_frame == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))
            elif self.num_frame == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))
            elif self.num_frame == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + segment_duration // 2

        return [(vid, sid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        images, activities = [], []

        for i, (vid, sid, fid) in enumerate(frames):
            fid = '{0:06d}'.format(fid)
            img = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
            img = self.transform(img)

            images.append(img)
            activities.append(self.anns[vid][sid]['group_activity'])

        images = torch.stack(images)
        activities = np.array(activities, dtype=np.int32)

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()

        return images, activities
