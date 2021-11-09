import torch
import torchvision.models as models

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    BackboneFinetuning,
    LearningRateMonitor,
)

from models.model import FakeVideoDetector, create_resnet
from models.efficientnet import PretrainedEfficientNet, get_effnet_transform
from utils.datasets import FakeVideoDataModule, get_resnet_transform


if __name__ == "__main__":

    # classifier = FakeVideoDetector(
    #     models.video.r3d_18(pretrained=True, progress=False),
    #     pos_weight=torch.tensor([1.93]),
    #     lr=1e-1,
    #     out_features=400,
    # )

    classifier = PretrainedEfficientNet(lr=1e-2)

    datamodule = FakeVideoDataModule(
        num_workers=20, batch_size=280, transforms=get_effnet_transform()
    )

    callbacks = [
        EarlyStopping(monitor="AUROC", mode="max", patience=20),
        BackboneFinetuning(5, should_align=True),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="effnet_{epoch}--{AUROC:.3f}",
            monitor="AUROC",
            mode="max",
        ),
        LearningRateMonitor(),
    ]

    trainer = Trainer(
        gpus=3,
        # gpus=[1, 2, 3],
        # strategy="ddp",
        callbacks=callbacks,
        log_every_n_steps=5,
        precision=16,
        # fast_dev_run=True,
    )

    trainer.fit(classifier, datamodule)
