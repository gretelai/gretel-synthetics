"""
Auxiliary datastructures for DGAN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgressInfo:
    """Information about DGAN training progress.

    Args:
      epoch: the current epoch, zero-based.
      total_epochs: the total number of epochs.
      batch: the current batch within the current epoch, zero-based.
      total_batches: the total number of batches in this epoch.
    """

    epoch: int
    total_epochs: int
    batch: int
    total_batches: int

    @property
    def frac_completed(self) -> float:
        """
        An estimation of which fraction of the overall task is completed.

        Returns:
            A number between 0.0 and 1.0 indicating which fraction of the task is completed.
        """
        return (
            self.epoch + 1 + float(self.batch + 1) / self.total_batches
        ) / self.total_epochs
