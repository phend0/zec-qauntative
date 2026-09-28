from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model on training data."""
        
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict distinct classes (-1, 0, 1)."""
        
    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
