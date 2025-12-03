import os
import math
import pandas as pd
import torchvision.transforms.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from src.Config import PreprocessingSettings

class DataProcessor:
    def __init__(self, df: pd.DataFrame, settings: PreprocessingSettings):
        self.df = df.copy()
        self.processedDir = Path(settings.processedDir)
        self.processedCsv = Path(settings.processedCsv)
        self.targetSize = tuple(settings.targetImageSize)
        self.augmentations = settings.augmentations

        self.processedDir.mkdir(parents=True, exist_ok=True)

        self.base_transform = transforms.Resize(max(self.targetSize)) 

    def clean_dataframe(self):
        print("Cleaning DataFrame...")
        self.df = self.df[self.df['file_path'].apply(lambda x: os.path.exists(x))]

        if 'difficulty' in self.df.columns:
            self.df['difficulty'] = self.df['difficulty'].str.lower().fillna('')
        if 'room' in self.df.columns:
            self.df['room'] = self.df['room'].str.lower().fillna('')
        if 'site' in self.df.columns:
            self.df['site'] = self.df['site'].astype(str).fillna('')

        self.df = self.df.fillna('')

    def process_and_augment(self):
        processed_data = []
        print("Processing and augmenting images...")

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            file_path = Path(row['file_path'])
            try:
                img = Image.open(file_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue

            img = self.base_transform(img)

            baseFilename = f"{file_path.stem}.jpg"
            savePath = self.processedDir / baseFilename
            img.save(savePath, format="JPEG")
            newRow = row.copy()
            newRow['file_path'] = str(savePath)
            processed_data.append(newRow)

            aug_images = self._apply_augmentations(img)
            for i, aug_img in enumerate(aug_images):
                augFilename = f"{file_path.stem}_aug{i}.jpg"
                augSavePath = self.processedDir / augFilename
                aug_img.save(augSavePath, format="JPEG")
                augRow = row.copy()
                augRow['file_path'] = str(augSavePath)
                processed_data.append(augRow)

        processedDf = pd.DataFrame(processed_data)
        processedDf.to_csv(self.processedCsv, index=False)
        print(f"Processed data saved to {self.processedDir}")
        print(f"CSV saved to {self.processedCsv}")

        return processedDf

    ## TODO think about applying multiple augmentations at once and fix traingles
    ## Maybe augments are not needed at all
    def _apply_augmentations(self, img: Image.Image):
        augmented = []

        if self.augmentations.get("flip", False):
            augmented.append(F.hflip(img))

        diag = int(math.ceil(math.sqrt(self.targetSize[0]**2 + self.targetSize[1]**2)))
        pad_amount_w = (diag - img.width) // 2
        pad_amount_h = (diag - img.height) // 2
        for angle in self.augmentations.get("rotate", []):
            padded = transforms.Pad((pad_amount_w, pad_amount_h), fill=(128,128,128))(img)
            rotated = F.rotate(padded, angle, expand=False, fill=(128,128,128))
            cropped = transforms.CenterCrop(self.targetSize)(rotated)
            augmented.append(cropped)

        for factor in self.augmentations.get("brightness", []):
            augmented.append(F.adjust_brightness(img, factor))

        for factor in self.augmentations.get("contrast", []):
            augmented.append(F.adjust_contrast(img, factor))

        return augmented

    def process(self):
        if self.processedCsv.exists():
            print(f"Processed CSV already exists at {self.processedCsv}. Loading instead of re-processing.")
            processed_df = pd.read_csv(self.processedCsv)
            return processed_df
        
        self.clean_dataframe()
        processed_df = self.process_and_augment()
        return processed_df