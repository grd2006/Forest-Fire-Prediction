"""
FastAPI endpoint for forest fire prediction and alerting.

Usage:
    python scripts/api.py
    
Then open: http://localhost:8000/docs
"""

import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(
    title="Forest Fire Prediction API",
    description="Real-time forest fire risk prediction and alerting",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = 'models/fire_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: Model not found at {MODEL_PATH}")
    model = None


# Request/Response schemas
class FirePredictionRequest(BaseModel):
    features: List[float]
    threshold: float = 0.5


class FirePredictionResponse(BaseModel):
    fire_probability: float
    alert: bool
    risk_level: str
    message: str


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Forest Fire Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=FirePredictionResponse)
async def predict(request: FirePredictionRequest):
    """
    Predict forest fire risk for given features.
    
    Returns:
    - fire_probability: confidence score (0-1)
    - alert: whether to trigger alert (True if prob >= threshold)
    - risk_level: 'LOW', 'MEDIUM', 'HIGH'
    - message: human-readable alert message
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.features) == 0:
        raise HTTPException(status_code=400, detail="Features list cannot be empty")
    
    # Convert to dataframe for prediction
    feature_data = pd.DataFrame([request.features])
    
    # Get prediction
    prediction = model.predict(feature_data)[0]
    probability = model.predict_proba(feature_data)[0, 1]
    
    # Determine alert and risk level
    alert = probability >= request.threshold
    
    if probability < 0.33:
        risk_level = "LOW"
        message = "✓ Fire risk is LOW. Normal conditions."
    elif probability < 0.67:
        risk_level = "MEDIUM"
        message = "⚠ Fire risk is MEDIUM. Monitor conditions."
    else:
        risk_level = "HIGH"
        message = "🔴 ALERT: Fire risk is HIGH. Immediate action recommended."
    
    return FirePredictionResponse(
        fire_probability=round(probability, 4),
        alert=alert,
        risk_level=risk_level,
        message=message
    )


@app.get("/batch-predict")
async def batch_predict(
    threshold: float = Query(0.5, ge=0, le=1, description="Alert threshold")
):
    """
    Batch predict on test dataset and return summary statistics.
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Load test data
    test_path = 'data/test.csv'
    if not os.path.exists(test_path):
        raise HTTPException(status_code=404, detail="Test data not found")
    
    test_df = pd.read_csv(test_path)
    X = test_df.drop(columns=['fire'])
    y_true = test_df['fire']
    
    # Predict
    y_pred_proba = model.predict_proba(X)[:, 1]
    alerts = (y_pred_proba >= threshold).astype(int)
    
    # Calculate stats
    n_total = len(y_true)
    n_alerts = alerts.sum()
    alert_rate = 100 * n_alerts / n_total if n_total > 0 else 0
    
    high_risk_indices = [i for i, p in enumerate(y_pred_proba) if p >= threshold]
    
    return {
        "total_records": n_total,
        "total_alerts": int(n_alerts),
        "alert_rate_percent": round(alert_rate, 2),
        "threshold": threshold,
        "high_risk_indices": high_risk_indices[:10],  # Top 10
        "message": f"Issued {n_alerts} fire risk alerts ({alert_rate:.1f}% alert rate)"
    }


if __name__ == '__main__':
    import uvicorn
    print("\n" + "=" * 60)
    print("Forest Fire Prediction API")
    print("=" * 60)
    print("Starting server at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host='0.0.0.0', port=8000)
