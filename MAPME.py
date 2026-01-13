import pandas as pd
import optuna
import bentoml
from src.utils.Config import Settings, settings
from src.processing.DataAnalyzer import DataAnalyzer
from src.processing.DataProcessor import DataProcessor
from src.processing.RoomDataLoader import RoomDataLoader
from src.training import train, tune
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.processing.RoomDataset import RoomDataModule
from src.models.Classifiers import RoomClassifier
from src.models.LightningModule import RoomLightningModule
from src.processing.RoomDataset import RoomDataModule
from src.models.Classifiers import RoomClassifier
from src.models.LightningModule import RoomLightningModule

# CSV_PATH = "data/processed/processed_dataset.csv"
CSV_PATH = "data/raw/dataset.csv"

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    dm = RoomDataModule(csv_path=CSV_PATH, batch_size=batch_size)
    dm.setup()

    model = RoomClassifier(num_classes=dm.num_classes)
    lit = RoomLightningModule(model, lr=lr)

    trainer = pl.Trainer(
        max_epochs=5,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(lit, datamodule=dm)

    return trainer.callback_metrics["val_loss"].item()

def tune():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

def train():
    dm = RoomDataModule(csv_path=CSV_PATH, batch_size=32, img_size=224)
    dm.setup()

    model = RoomClassifier(num_classes=dm.num_classes)
    lit = RoomLightningModule(model, lr=1e-3)

    logger = WandbLogger(project="room-classifier")

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint],
    )

    trainer.fit(lit, datamodule=dm)

    best_model_path = checkpoint.best_model_path
    lit_model = RoomLightningModule.load_from_checkpoint(
        best_model_path, 
        model=RoomClassifier(num_classes=dm.num_classes)
    )
    
    model_to_save = lit_model.model

    class_names = dm.train_set.dataset.class_names
    print(f"Class names: {class_names}")
    
    bentoml.pytorch.save_model(
        "room_classifier",
        model_to_save,
        signatures={"__call__": {"batchable": True}},
        metadata={
            "class_names": class_names,
            "img_size": dm.img_size
        }
    )
    print(f"Model saved to BentoML store from {best_model_path}")

def loadSettings():
    return settings

def loadData(settings: Settings) -> pd.DataFrame:
    dataLoader = RoomDataLoader(settings)
    data = dataLoader.loadData()
    return data
    
def analyzeData(data: pd.DataFrame, settings: Settings):
    analyzer = DataAnalyzer(data, settings)
    return analyzer.analyze()

def main():
    settings = loadSettings()
    print("Current Configuration:")
    print(settings.describe())
    data = loadData(settings)
    preprocessingSettings = analyzeData(data, settings)
    print("Preprocessing Settings:")
    print(preprocessingSettings.describe())

    dataProcessor = DataProcessor(data, preprocessingSettings)
    processed_data = dataProcessor.process()
    print(processed_data.head())

if __name__ == "__main__":
    main()
    train()
    # tune()