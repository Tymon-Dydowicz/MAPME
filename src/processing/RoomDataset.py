import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image


class RoomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.paths = self.df["file_path"].tolist()
        self.labels = self.df["room"].astype("category").cat.codes.tolist()
        self.transform = transform

        self.class_names = (
            self.df["room"].astype("category").cat.categories.tolist()
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class RoomDataModule(LightningDataModule):
    def __init__(self, csv_path, batch_size=32, img_size=224):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        dataset = RoomDataset(self.csv_path, transform=self.transform)

        n = len(dataset)
        n_train = int(0.8 * n)
        n_val = n - n_train

        self.train_set, self.val_set = random_split(dataset, [n_train, n_val])

        self.num_classes = len(dataset.class_names)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)