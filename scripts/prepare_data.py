"""
Prepare data for training: preprocess forestfires.csv and save train/val/test splits.

Usage:
    python scripts/prepare_data.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import preprocess_and_split


def main():
    input_file = 'data/forestfires.csv'
    output_dir = 'data'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please download it first.")
        return
    
    print("=" * 60)
    print("Forest Fire Data Preparation")
    print("=" * 60)
    
    train_df, val_df, test_df, scaler = preprocess_and_split(
        input_file,
        output_dir=output_dir,
        test_size=0.2,
        val_size=0.1
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
