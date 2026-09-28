import logging
import pandas as pd
from data_loader import DataLoader
from features import FeatureEngineer
from target import TargetGenerator
from validation import PurgedKFold
from models.tabular import LogRegModel, LightGBMModel
from models.sequence import PyTorchSequenceModel
from backtest import Backtester
from config import config

# Set root level explicitly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing ZEC Predictive ML Pipeline...")
    
    # 1. robust ingestion
    dl = DataLoader()
    df = dl.load_and_align()
    
    # 2. feature engineering
    fe = FeatureEngineer(df)
    df = fe.generate_features()
    FeatureEngineer.check_stationarity(df)
    
    # 3. target labeling
    tg = TargetGenerator(df)
    df = tg.generate_targets()
    
    # Clean and split targets from features
    # Exclude open, high, low, close, volume explicitly to prevent leakage (we only want engineered indicators)
    drop_cols = ['label'] + [col for col in df.columns if any(p in col for p in ['_open', '_high', '_low', '_close', '_volume']) and 'log_return' not in col]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['label']
    
    logger.info(f"Feature matrix strictly prepped. Total columns: {len(X.columns)}")
    
    # 4 & 5. Embargoed Validation and Model Orchestration
    pkf = PurgedKFold(n_splits=3)
    
    # Can toggle between LogRegModel(), LightGBMModel(use_optuna=True), PyTorchSequenceModel()
    model = LightGBMModel(use_optuna=False)
    
    # Default everything to cash (0)
    predictions = pd.Series(index=X.index, data=0)
    
    logger.info("Initiating Purged & Embargoed Walk-Forward Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X, y)):
        logger.info(f"--- Processing Fold {fold + 1} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        
        predictions.iloc[test_idx] = preds
        
    # 6. Realistic Backtesting Simulator
    bt = Backtester(df, predictions.values)
    bt.run()

if __name__ == "__main__":
    main()
