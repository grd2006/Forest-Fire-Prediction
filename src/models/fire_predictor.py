"""
Machine learning models for forest fire prediction.
Includes RandomForest and XGBoost baselines.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
import xgboost as xgb
import joblib


class FirePredictor:
    """Base predictor class."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.metrics = {}
    
    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Compute metrics on test set."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        return self.metrics
    
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"✓ Saved model to {path}")
    
    def load(self, path):
        self.model = joblib.load(path)
        print(f"✓ Loaded model from {path}")


class RandomForestPredictor(FirePredictor):
    """Random Forest classifier."""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train, **kwargs):
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"✓ {self.name} training complete.")


class XGBoostPredictor(FirePredictor):
    """XGBoost classifier."""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        super().__init__("XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def train(self, X_train, y_train, eval_set=None, **kwargs):
        print(f"Training {self.name}...")
        if eval_set is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        print(f"✓ {self.name} training complete.")


def get_model(model_name, **kwargs):
    """Factory function to get a model by name."""
    models = {
        'rf': RandomForestPredictor,
        'xgboost': XGBoostPredictor,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)
