from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model on training data."""
        pass
        
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict distinct classes (-1, 0, 1)."""
        pass
        
    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        pass
