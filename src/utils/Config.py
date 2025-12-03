import json
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Any, Dict, List

class Settings(BaseSettings):
    debug: bool = Field(False, env="DEBUG", validation_alias="DEBUG")
    imageExtension: List[str] = Field([".png", ".jpg"], env="IMAGE_EXTENSION", validation_alias="IMAGE_EXTENSION")
    imageNamingConvention: str = Field("*{index}_{site}_{room}_{difficulty}", env="IMAGE_NAMING_CONVENTION", validation_alias="IMAGE_NAMING_CONVENTION")
    difficultyMode: str = Field("OFF", env="DIFFICULTY_MODE", validation_alias="DIFFICULTY_MODE")
    difficultyOrder: List[str] = Field(["EASY", "MEDIUM", "HARD"], env="DIFFICULTY_ORDER", validation_alias="DIFFICULTY_ORDER")
    dataPath: str = Field("./data/raw", env="DATA_PATH", validation_alias="DATA_PATH")
    dataCsv: str = Field("./data/raw/dataset.csv", env="DATA_CSV", validation_alias="DATA_CSV")

    class Config:
        env_file = "config.env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def describe(self) -> str:
        return (
            f"Debug: {self.debug}\n"
            f"Image Extensions: {self.imageExtension}\n"
            f"Image Naming Convention: {self.imageNamingConvention}\n"
            f"Difficulty Mode: {self.difficultyMode}\n"
            f"Difficulty Order: {self.difficultyOrder}\n"
            f"Data Path: {self.dataPath}\n"
            f"Data CSV: {self.dataCsv}\n"
        )
    
class PreprocessingSettings(BaseSettings):
    adversarialTraining: bool = Field(False, env="ADVERSARIAL_TRAINING", validation_alias="ADVERSARIAL_TRAINING")
    oversampling: bool = Field(False, env="OVERSAMPLING", validation_alias="OVERSAMPLING")
    processedDir: str = Field("./data/processed", env="PROCESSED_DIR", validation_alias="PROCESSED_DIR")
    processedCsv: str = Field("./data/processed/processed_dataset.csv", env="PROCESSED_CSV", validation_alias="PROCESSED_CSV")
    targetImageSize: List[int] = Field([224, 224], env="TARGET_IMAGE_SIZE", validation_alias="TARGET_IMAGE_SIZE")
    augmentations: Dict[str, Any] = Field(
        default_factory=lambda: {"flip": True, "rotate": [15, -15], "brightness": [0.9, 1.1], "contrast": [0.9, 1.1]},
        env="AUGMENTATIONS"
    )

    def __init__(self, **values):
        super().__init__(**values)
        if isinstance(self.augmentations, str):
            try:
                self.augmentations = json.loads(self.augmentations)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse AUGMENTATIONS: {self.augmentations}")

    class Config:
        env_file = "config.env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def describe(self) -> str:
        return (
            f"Adversarial Training: {self.adversarialTraining}\n"
            f"Oversampling: {self.oversampling}\n"
        )

settings = Settings()