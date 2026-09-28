import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import random
from sklearn.preprocessing import RobustScaler
import os

from .base import BaseModel
from config import config

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Enforce reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int = 30):
        self.X = X
        self.y = y
        self.window_size = window_size
        
    def __len__(self):
        return len(self.X) - self.window_size + 1
        
    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.window_size]
        y_val = self.y[idx + self.window_size - 1]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, rnn_type='GRU'):
        super().__init__()
        self.rnn_type = rnn_type
        
        # PyTorch restricts dropout > 0 if num_layers == 1
        eff_dropout = dropout if num_layers > 1 else 0
        
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=eff_dropout)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=eff_dropout)
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 3) # Output bounds: (-1, 0, 1) mapped to (0, 1, 2)
        
    def forward(self, x):
        if self.rnn_type == 'LSTM':
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
        
        # Extract features from the final sequence step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class PyTorchSequenceModel(BaseModel):
    def __init__(self, window_size=30, rnn_type='GRU'):
        self.window_size = window_size
        self.rnn_type = rnn_type
        self.scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(config.random_seed)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info(f"Training {self.rnn_type} model on {self.device}...")
        
        # Scaling must occur purely on the training window
        X_scaled = self.scaler.fit_transform(X_train.values)
        y_mapped = y_train.values + 1 # shift target to 0, 1, 2 classes
        
        # Split train strictly for early stopping checks (temporal split)
        split_idx = int(len(X_scaled) * 0.8)
        X_t, y_t = X_scaled[:split_idx], y_mapped[:split_idx]
        X_v, y_v = X_scaled[split_idx:], y_mapped[split_idx:]
        
        train_dataset = TimeSeriesDataset(X_t, y_t, self.window_size)
        val_dataset = TimeSeriesDataset(X_v, y_v, self.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        self.model = RNNModel(
            input_size=X_scaled.shape[1],
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            rnn_type=self.rnn_type
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
            val_loss /= max(1, len(val_loader))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_rnn.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        if os.path.exists('best_rnn.pth'):
            self.model.load_state_dict(torch.load('best_rnn.pth'))
        self.model.eval()

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test.values)
        
        # Prepend zero-padding to allow predicting the very first elements in X_test
        # Since Dataset consumes `window_size` points to make 1 prediction
        pad = np.zeros((self.window_size - 1, X_scaled.shape[1]))
        X_padded = np.vstack([pad, X_scaled])
        
        y_dummy = np.zeros(len(X_padded))
        test_dataset = TimeSeriesDataset(X_padded, y_dummy, self.window_size)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                
        return np.array(probs)
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_test)
        # Shift back from (0, 1, 2) to (-1, 0, 1)
        return np.argmax(probs, axis=1) - 1
