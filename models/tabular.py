import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import optuna
import shap
import logging
from typing import Dict, Any

from .base import BaseModel
from config import config

logger = logging.getLogger(__name__)

class LogRegModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', LogisticRegression(
                penalty='l2', 
                class_weight='balanced', 
                random_state=config.random_seed,
                max_iter=1000,
                **kwargs
            ))
        ])
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info("Training Class-Weighted Logistic Regression...")
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X_test)


class LightGBMModel(BaseModel):
    def __init__(self, use_optuna: bool = False):
        self.use_optuna = use_optuna
        self.model = None
        self.best_params = None
        
    def _optimize(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Simple temporal train/val split for Optuna inner loop
        split_idx = int(len(X_train) * 0.8)
        X_t, y_t = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
        X_v, y_v = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'class_weight': 'balanced',
                'n_estimators': trial.suggest_categorical('n_estimators', config.grids.lightgbm_grid['n_estimators']),
                'learning_rate': trial.suggest_categorical('learning_rate', config.grids.lightgbm_grid['learning_rate']),
                'max_depth': trial.suggest_categorical('max_depth', config.grids.lightgbm_grid['max_depth']),
                'subsample': trial.suggest_categorical('subsample', config.grids.lightgbm_grid['subsample']),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', config.grids.lightgbm_grid['colsample_bytree']),
                'random_state': config.random_seed,
                'verbose': -1
            }
            clf = lgb.LGBMClassifier(**params)
            
            # Shift labels for LightGBM multiclass: -1, 0, 1 -> 0, 1, 2
            y_t_mapped = y_t + 1
            y_v_mapped = y_v + 1
            
            clf.fit(X_t, y_t_mapped, eval_set=[(X_v, y_v_mapped)])
            preds = clf.predict(X_v)
            acc = np.mean(preds == y_v_mapped)
            return acc
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        logger.info(f"Best Optuna parameters: {study.best_params}")
        return study.best_params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.use_optuna:
            logger.info("Running Optuna optimization for LightGBM...")
            self.best_params = self._optimize(X_train, y_train)
        else:
            logger.info("Training LightGBM with default grid parameters...")
            self.best_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'class_weight': 'balanced',
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'random_state': config.random_seed,
                'verbose': -1
            }
            
        self.model = lgb.LGBMClassifier(**self.best_params)
        # Shift labels for LightGBM: -1, 0, 1 -> 0, 1, 2
        self.model.fit(X_train, y_train + 1)
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X_test)
        # Re-shift back to -1, 0, 1
        return preds - 1
        
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X_test)
        
    def explain(self, X: pd.DataFrame):
        """Extract TreeSHAP global summary."""
        logger.info("Generating TreeSHAP explanations...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values
