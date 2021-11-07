import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from torchvision.transforms import Compose, Resize
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Div255
from sklearn.model_selection import train_test_split

from utils.config import TRAIN_FOLDER, TRAIN_LABELS_FILE, TEST_FOLDER, SEED

pl.seed_everything(SEED)


def get_videofiles_and_labels():
    fname_label_pairs = []
    with open(TRAIN_LABELS_FILE, "r") as f:
        for line in f:
            try:
                fname, label = line.strip().split(",")
                fname = os.path.join(TRAIN_FOLDER, fname)
                label = {"label": torch.tensor([int(label)], dtype=torch.float32)}
                fname_label_pairs.append(tuple((fname, label)))
            except:
                pass
    return fname_label_pairs


class FakeVideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=16,
        num_workers=12,
        frames_per_video=16,
        framesize=256,
        vid_duration=5.0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames_per_video = frames_per_video
        self.framesize = framesize
        self.vid_duration = vid_duration

        self.train_vids, self.val_vids = train_test_split(
            get_videofiles_and_labels(),
            test_size=0.1,
            train_size=0.9,
            shuffle=True,
            random_state=SEED,
        )
        # self.test_vids = [
        #     os.path.join(TEST_FOLDER, fname) for fname in os.listdir(TEST_FOLDER)
        # ]

        self.transforms = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.frames_per_video),
                            Resize(self.framesize),
                            Div255(),
                        ]
                    ),
                )
            ]
        )

    def create_dataloader(self, vids):
        dataset = LabeledVideoDataset(
            vids,
            UniformClipSampler(self.vid_duration),
            transform=self.transforms,
            decode_audio=False,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_vids)

    def val_dataloader(self):
        return self.create_dataloader(self.val_vids)

