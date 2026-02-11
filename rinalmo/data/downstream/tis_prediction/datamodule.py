from torch.utils.data import DataLoader

import pytorch_lightning as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.tis_prediction.dataset import (
    TISDataset,
    tis_collate_fn,
)


class TISDataModule(pl.LightningDataModule):
    """Lightning DataModule for TIS prediction.

    Expects a directory layout::

        data_root/
            train.csv
            val.csv
        test_data_root/
            test.csv

    Each CSV must have columns ``sequence`` and ``tis_positions`` (see
    :class:`TISDataset` for format details).

    Training data is pre-filtered to contain only windows with at least
    one TIS site (positive-window mining), so a standard shuffled
    DataLoader is sufficient — no structured batch sampler needed.
    """

    def __init__(
        self,
        data_root: Optional[Union[Path, str]] = None,
        test_data_root: Optional[Union[Path, str]] = None,
        alphabet: Alphabet = Alphabet(),
        max_seq_len: int = 1022,
        target_block_size: int = 400,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_root = Path(data_root) if data_root else None
        self.test_data_root = Path(test_data_root) if test_data_root else None
        self.alphabet = alphabet
        self.max_seq_len = max_seq_len
        self.target_block_size = target_block_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        if self.data_root is not None:
            self.train_dataset = TISDataset(
                self.data_root / "train.csv",
                alphabet=self.alphabet,
                max_seq_len=self.max_seq_len,
                target_block_size=self.target_block_size,
            )
            self.val_dataset = TISDataset(
                self.data_root / "val.csv",
                alphabet=self.alphabet,
                max_seq_len=self.max_seq_len,
                target_block_size=self.target_block_size,
            )

        if self.test_data_root is not None:
            self.test_dataset = TISDataset(
                self.test_data_root / "test.csv",
                alphabet=self.alphabet,
                max_seq_len=self.max_seq_len,
                target_block_size=self.target_block_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=tis_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=tis_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=tis_collate_fn,
        )
