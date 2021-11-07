import torch
import torchvision.models as models

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.model import FakeVideoDetector, create_resnet, PretrainedDetector
from utils.datasets import FakeVideoDataModule, get_resnet_transform


if __name__ == "__main__":

    classifier = FakeVideoDetector(
        models.video.r3d_18(pretrained=True, progress=False),
        pos_weight=torch.tensor([1.93]),
        lr=1e-1,
        out_features=400,
    )

    datamodule = FakeVideoDataModule(
        num_workers=40, batch_size=72, transforms=get_resnet_transform()
    )

    callbacks = [
        EarlyStopping(monitor="AUROC", mode="max", patience=20),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}--{AUROC:.3f}",
            monitor="AUROC",
            mode="max",
        ),
    ]

    trainer = Trainer(
        #         gpus=1,
        gpus=[1, 2, 3],
        strategy="ddp",
        callbacks=callbacks,
        log_every_n_steps=5,
        precision=16,
        # fast_dev_run=True,
    )

    trainer.fit(classifier, datamodule)
