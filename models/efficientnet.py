import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)
from torchmetrics import MetricCollection, AUROC, Accuracy


class PretrainedEfficientNet(pl.LightningModule):
    def __init__(self, pos_weight=1.93, lr=1e-2):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight

        effnet = models.efficientnet_b0(pretrained=True)

        self.backbone = nn.Sequential(*list(effnet.children())[:-1])

        in_features = effnet.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True), nn.Linear(in_features, 1)
        )

        self.metrics = MetricCollection([Accuracy(), AUROC(),])

        self.loss = nn.BCEWithLogitsLoss(torch.tensor([self.pos_weight]))
        self.sigmoid = nn.Sigmoid()

        self.save_hyperparameters()

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, H, W, C)
        # but provided by datamodule is (B, C, T, H, W)
        y_hat = self(batch["video"].squeeze())

        loss = self.loss(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch["video"].squeeze())
        labels = batch["label"]

        loss = self.loss(y_hat, labels)
        probas = self.sigmoid(y_hat)

        self.metrics.update(probas, labels.type(torch.int32))
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        self.log_dict(self.metrics.compute())

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "AUROC",
        }


def get_effnet_transform(side_size=224,):

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [UniformTemporalSubsample(1), ShortSideScale(size=side_size),]
        ),
    )
