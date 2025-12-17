"""
Train forest fire prediction model using preprocessed data.

Usage:
    python scripts/train.py
    
Trains a RandomForest classifier with class weighting and probability calibration,
then evaluates on validation and test sets.
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             precision_score, recall_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(data_dir='data'):
    """Load preprocessed train/val/test splits."""
    train_file = os.path.join(data_dir, 'train.csv')
    val_file = os.path.join(data_dir, 'val.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        print("Error: Please run 'python scripts/prepare_data.py' first to create splits.")
        return None
    
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    
    return train, val, test


def train_model(X_train, y_train, X_val, y_val, n_estimators=300):
    """Train RandomForest with class weighting."""
    print("Training RandomForest with class_weight='balanced'...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Validate on validation set
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"[OK] Validation accuracy: {val_acc:.4f}")
    
    return clf


def calibrate_model(clf, X_val, y_val):
    """Calibrate model predictions using validation set."""
    print("Calibrating model with isotonic method...")
    calibrator = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
    calibrator.fit(X_val, y_val)
    print("[OK] Model calibrated")
    
    return calibrator


def find_best_threshold(model, X_val, y_val):
    """Find best threshold on validation set to maximize accuracy."""
    val_probs = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_acc = -1
    
    print("Searching for optimal threshold...")
    for t in np.linspace(0.01, 0.99, 99):
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t
    
    print(f"[OK] Best threshold: {best_threshold:.4f} (validation accuracy: {best_acc:.4f})")
    return best_threshold


def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate model on test set."""
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    
    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_test, test_preds)),
        'roc_auc': float(roc_auc_score(y_test, test_probs)),
        'f1': float(f1_score(y_test, test_preds)),
        'precision': float(precision_score(y_test, test_preds)),
        'recall': float(recall_score(y_test, test_preds)),
        'confusion_matrix': confusion_matrix(y_test, test_preds).tolist()
    }
    
    return metrics, test_preds


def print_metrics(metrics):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Threshold:     {metrics['threshold']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4%}")
    print(f"Precision:     {metrics['precision']:.4%}")
    print(f"Recall:        {metrics['recall']:.4%}")
    print(f"F1 Score:      {metrics['f1']:.4%}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4%}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:3d} | False Positives: {cm[0][1]:3d}")
    print(f"  False Negatives: {cm[1][0]:3d} | True Positives:  {cm[1][1]:3d}")
    print("=" * 60 + "\n")


def main(args):
    print("=" * 60)
    print("FOREST FIRE PREDICTION MODEL TRAINING")
    print("=" * 60 + "\n")
    
    # Load data
    print("Loading preprocessed data...")
    data = load_data(args.data_dir)
    if data is None:
        return
    
    train, val, test = data
    X_train = train.drop(columns=['fire'])
    y_train = train['fire']
    X_val = val.drop(columns=['fire'])
    y_val = val['fire']
    X_test = test.drop(columns=['fire'])
    y_test = test['fire']
    
    print(f"[OK] Loaded train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}\n")
    
    # Train model
    clf = train_model(X_train, y_train, X_val, y_val, args.n_estimators)
    
    # Calibrate
    print()
    calibrator = calibrate_model(clf, X_val, y_val)
    
    # Find best threshold
    print()
    best_threshold = find_best_threshold(calibrator, X_val, y_val)
    
    # Evaluate on test
    print()
    metrics, test_preds = evaluate_model(calibrator, X_test, y_test, best_threshold)
    print_metrics(metrics)
    
    # Save model and metrics
    model_path = os.path.join(args.model_dir, args.model_name)
    metrics_path = os.path.join(args.model_dir, args.metrics_name)
    
    joblib.dump(calibrator, model_path)
    print(f"[OK] Model saved to {model_path}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forest fire prediction model')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory with preprocessed train/val/test splits')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save trained model')
    parser.add_argument('--model-name', type=str, default='fire_model_combined_rf_calibrated.pkl',
                        help='Name of model file to save')
    parser.add_argument('--metrics-name', type=str, default='fire_model_combined_rf_calibrated_metrics.json',
                        help='Name of metrics file to save')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Number of trees in RandomForest')
    
    args = parser.parse_args()
    main(args)
