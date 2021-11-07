from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.model import FakeVideoDetector, create_resnet
from utils.datasets import FakeVideoDataModule


if __name__ == "__main__":

    classifier = FakeVideoDetector()
    datamodule = FakeVideoDataModule(num_workers=10, batch_size=16)
    callbacks = [
                EarlyStopping(monitor="AUROC", mode="max", patience=20),
                ModelCheckpoint(
                    dirpath='checkpoints',
                    filename='{epoch}--{AUROC:.3f}', monitor="AUROC", mode="max",
                ),
            ]

    trainer = Trainer(
#         gpus=1,
        gpus=[1,2,3],
        strategy='ddp',
        callbacks=callbacks,
        log_every_n_steps=5,
#         precision=16,
#         fast_dev_run=True,
    )

    trainer.fit(classifier, datamodule)