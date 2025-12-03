import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.processing.RoomDataset import RoomDataModule
from src.models.Classifiers import RoomClassifier
from src.models.LightningModule import RoomLightningModule

CSV_PATH = "data/processed/processed.csv"

def main():
    dm = RoomDataModule(csv_path=CSV_PATH, batch_size=32, img_size=224)
    dm.setup()

    model = RoomClassifier(num_classes=dm.num_classes)
    lit = RoomLightningModule(model, lr=1e-3)

    logger = WandbLogger(project="room-classifier")

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint],
    )

    trainer.fit(lit, datamodule=dm)