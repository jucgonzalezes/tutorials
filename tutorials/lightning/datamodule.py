import pytorch_lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        """
        Creates the custom dataset by loading a dataloader and splitting
        it into train, test, and validation.
        """
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )

        self.train_dataset, self.validation_dataset = random_split(
            entire_dataset, [50_000, 10_000]
        )

        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def download_data(self):
        """
        Downloads the data and saves them to self.data_dir. In this
        particular example we are downloading the MNIST dataset for
        both train and test tasks.
        """
        datasets.MNIST(self.data_dir, train=True)
        datasets.MNIST(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            shuffle=True,
        )

    def validation_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            shuffle=False,
        )
