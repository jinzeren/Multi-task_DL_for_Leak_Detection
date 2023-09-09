from typing import Any, Dict, Optional, Tuple
import numpy as np
import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class LeakDetectionDataset(Dataset):

    def __init__(self, data_dir: str, data_denoised: bool = True, purpose: str = 'train', transfer: bool = False):
        if data_denoised:
            file_suffix = 'd'
        else:
            file_suffix = 'o'
        assert purpose in ['train', 'val', 'test']
        if transfer:
            insert_part = 'transfer_'
        else:
            insert_part = ''
        self.input = torch.from_numpy(np.load(os.path.join(
            data_dir, f"mel_{insert_part}{purpose}_{file_suffix}.npy"))).float()
        self.output = torch.from_numpy(np.load(os.path.join(
            data_dir, f"output_{insert_part}{purpose}_{file_suffix}.npy"))).float()
        # if purpose in ['train', 'val']:
        #     self.input = torch.from_numpy(np.load(os.path.join(
        #         data_dir, f"mel_{insert_part}{purpose}_{file_suffix}.npy"))).float()
        #     self.output = torch.from_numpy(np.load(os.path.join(
        #         data_dir, f"output_{insert_part}{purpose}_{file_suffix}.npy"))).float()
        # else:
        #     self.input = torch.from_numpy(np.load(os.path.join(
        #         data_dir, f"mel_{insert_part}{purpose}_d.npy"))).float()
        #     self.output = torch.from_numpy(np.load(os.path.join(
        #         data_dir, f"output_{insert_part}{purpose}_d.npy"))).float()

    def __getitem__(self, idx):
        return self.input[idx].unsqueeze(0), self.output[idx]

    def __len__(self):
        return len(self.input)


class LeakDetectionModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_denoised: bool = True,
        transfer: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = LeakDetectionDataset(
            self.hparams.data_dir, self.hparams.data_denoised, 'train', self.hparams.transfer)
        self.data_val = LeakDetectionDataset(
            self.hparams.data_dir, self.hparams.data_denoised, 'val', self.hparams.transfer)
        self.data_test = LeakDetectionDataset(
            self.hparams.data_dir, self.hparams.data_denoised, 'test', self.hparams.transfer)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = LeakDetectionModule()
