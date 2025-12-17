"""
Generate predictions and fire alerts on new data.

Usage:
    python scripts/predict.py --model models/fire_model.pkl --input data/test.csv --threshold 0.5
"""

import sys
import os
import argparse
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(args):
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    
    model = joblib.load(args.model)
    print(f"[OK] Loaded model from {args.model}")
    
    # Load input data
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
    
    df = pd.read_csv(args.input)
    print(f"[OK] Loaded {df.shape[0]} records from {args.input}")
    
    # Separate features and target (if exists)
    has_target = 'fire' in df.columns
    if has_target:
        X = df.drop(columns=['fire'])
        y_true = df['fire']
    else:
        X = df
        y_true = None
    
    # Generate predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': y_pred,
        'fire_probability': y_pred_proba,
        'alert': (y_pred_proba >= args.threshold).astype(int)
    })
    
    # Add actual target if available
    if has_target:
        results['actual'] = y_true.values
        results['correct'] = (results['prediction'] == results['actual']).astype(int)
        accuracy = results['correct'].mean()
        print(f"\nAccuracy: {accuracy:.4f}")
    
    # Save results
    output_file = args.output or args.input.replace('.csv', '_predictions.csv')
    results.to_csv(output_file, index=False)
    print(f"[OK] Predictions saved to {output_file}")
    
    # Print alert summary
    n_alerts = results['alert'].sum()
    alert_rate = 100 * n_alerts / len(results)
    print(f"\n" + "=" * 60)
    print("ALERT SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(results)}")
    print(f"High-risk alerts (prob >= {args.threshold}): {n_alerts} ({alert_rate:.1f}%)")
    
    # Show top fire-risk records
    top_risks = results.nlargest(5, 'fire_probability')
    print(f"\nTop 5 high-risk records:")
    print(top_risks[['fire_probability', 'alert']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions and fire alerts.')
    parser.add_argument('--model', default='models/fire_model.pkl',
                        help='Trained model path (default: models/fire_model.pkl)')
    parser.add_argument('--input', default='data/test.csv',
                        help='Input CSV file with features (default: data/test.csv)')
    parser.add_argument('--output', default=None,
                        help='Output CSV file for predictions (default: input_predictions.csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Alert threshold: probability >= threshold triggers alert (default: 0.5)')
    
    args = parser.parse_args()
    main(args)
