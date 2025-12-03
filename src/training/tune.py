import optuna
import pytorch_lightning as pl

from src.processing.RoomDataset import RoomDataModule
from src.models.Classifiers import RoomClassifier
from src.models.LightningModule import RoomLightningModule

CSV_PATH = "data/processed/processed.csv"

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

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)