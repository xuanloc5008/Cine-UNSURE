from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class H5NoisyVolumeDataset(Dataset[torch.Tensor]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as h5:
            if "y" not in h5:
                raise KeyError(f"{self.path} must contain dataset 'y'")
            self.length = int(h5["y"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        with h5py.File(self.path, "r") as h5:
            arr = h5["y"][index]
        return torch.from_numpy(arr).float()
