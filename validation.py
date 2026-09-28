import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

class PurgedKFold:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        # Embargo prevents the right side of the train set from leaking into the test set
        self.embargo = config.target.vertical_barrier_periods

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for purged and embargoed walk-forward Cross-Validation.
        In an expanding window (walk-forward) setup, test data is always *after* train data.
        Therefore, we insert an 'embargo' period equal to the target label horizon
        between the end of the train set and the start of the test set.
        """
        n = len(X)
        indices = np.arange(n)
        test_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end + self.embargo
            test_end = test_start + test_size
            
            test_end = min(test_end, n)
                
            if test_start >= n:
                logger.warning(f"Fold {i+1}: Not enough data for embargoed test set.")
                break
                
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            
            logger.info(f"Fold {i+1}: Train: {len(train_idx)} | Embargo: {self.embargo} | Test: {len(test_idx)}")
            yield train_idx, test_idx
