import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import MetricCollection, AUROC, Accuracy
from pytorchvideo.models import resnet


def create_resnet(model_depth=50):
    return resnet.create_resnet(
        input_channel=3,
        model_depth=model_depth,
        model_num_class=1,
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )


class FakeVideoDetector(pl.LightningModule):
    def __init__(self, model=create_resnet(), pos_weight=None, lr=0.01):
        super().__init__()

        self.lr = lr

        self.model = model
        self.metrics = MetricCollection([Accuracy(), AUROC(),])

        self.loss = nn.BCEWithLogitsLoss(pos_weight)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self(batch["video"])

        loss = self.loss(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch["video"])
        labels = batch["label"]
        loss = self.loss(y_hat, labels)
        self.metrics.update(y_hat, labels.type(torch.int32))
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        self.log_dict(self.metrics.compute())

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PretrainedDetector(FakeVideoDetector):

    def __init__(self, model, pos_weight=None, out_features=None):
        super().__init__(model=model, pos_weight=pos_weight)

        if out_features:
            self.out_layer = nn.Linear(out_features, 1)
        else:
            self.out_layer = nn.Linear(self.model.fc.out_features, 1)
        

    def forward(self, x):
        x = self.model(x)
        x = self.out_layer(x)
        return x