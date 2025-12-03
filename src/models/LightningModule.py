import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule

class RoomLightningModule(LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        self.train_acc = Accuracy(task="multiclass", num_classes=5)
        self.val_acc = Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage):
        imgs, labels = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = self.train_acc(preds, labels) if stage == "train" else self.val_acc(preds, labels)

        self.log(f"{stage}_loss", loss, on_epoch=True)
        self.log(f"{stage}_acc", acc, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)