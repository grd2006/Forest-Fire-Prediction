# Forest Fire Prediction Project - Setup & Run Guide

## Dependencies

The project requires the following Python packages (see `requirements.txt`):

```
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.3
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
ipython==8.18.1
joblib==1.3.2
fastapi==0.104.1
uvicorn==0.24.0
pytest==7.4.3
```

## Installation

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Running the Project

Follow these steps in order to prepare data, train the model, and generate predictions:

### Step 1: Prepare Data
```
python scripts/prepare_data.py
```
**Expected output:** 
- `data/train.csv` (training dataset)
- `data/val.csv` (validation dataset)
- `data/test.csv` (test dataset)

✓ **Verify:** Check that all three CSV files are created in the `data/` directory.

### Step 2: Train Model
```
python scripts/train.py
```
**Expected output:**
- `models/fire_model.pkl` (trained model)
- `models/fire_model_combined_rf_calibrated_metrics.json` (model metrics)

✓ **Verify:** Check that model files are created in the `models/` directory.

### Step 3: Generate Predictions
```
python scripts/predict.py --model models/fire_model_combined_rf_calibrated.pkl --input data/test.csv
```
**Expected output:**
- `data/test_predictions.csv` (predictions on test data)

✓ **Verify:** Check that `test_predictions.csv` is created in the `data/` directory.

### Step 4: Run Web Application (Optional)
Once all steps complete successfully, start the web app:
```
python -m streamlit run app.py
```

The web application will be available at `http://localhost:8501`

## CSV Files Verification Checklist

After completing each step, verify the following CSV files are created:

| File | Created After Step | Location |
|------|------------------|----------|
| `train.csv` | Prepare Data (Step 1) | `data/` |
| `val.csv` | Prepare Data (Step 1) | `data/` |
| `test.csv` | Prepare Data (Step 1) | `data/` |
| `test_predictions.csv` | Generate Predictions (Step 3) | `data/` |

## Troubleshooting

- If `forestfires.csv` is not found, ensure it exists in the `data/` directory before running Step 1
- If any step fails, check the error message and ensure all dependencies are correctly installed
- All scripts must be run from the project root directory
 
