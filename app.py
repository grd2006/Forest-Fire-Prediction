import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Forest Fire Model", layout="wide")

st.title("🔥 Forest Fire Prediction Model")
st.write("Real-time forest fire prediction using machine learning")

# Load data
df = pd.read_csv("data/test_predictions.csv")

# Sidebar
st.sidebar.header("Model Information")
try:
    with open("models/fire_model_combined_rf_calibrated_metrics.json") as f:
        metrics = json.load(f)
    st.sidebar.write("**Model Metrics:**")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            st.sidebar.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
except:
    st.sidebar.info("Metrics file not found")

# Main content
col1, col2, col3 = st.columns(3)
with col1:
    accuracy = (df['correct'].sum() / len(df)) * 100
    st.metric("Accuracy", f"{accuracy:.2f}%")
with col2:
    fire_count = (df['prediction'] == 1).sum()
    st.metric("Fire Predictions", fire_count)
with col3:
    actual_fires = (df['actual'] == 1).sum()
    st.metric("Actual Fires", actual_fires)

st.divider()

# Confusion Matrix and Prediction Accuracy Pie Chart
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['actual'], df['prediction'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'],
                ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    st.pyplot(fig)

with col2:
    st.subheader("Prediction Accuracy")
    correct = df['correct'].sum()
    incorrect = len(df) - correct
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie([correct, incorrect], 
                                        labels=['Correct', 'Incorrect'],
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        startangle=90,
                                        textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    ax.set_title('Prediction Results', fontsize=14, fontweight='bold')
    st.pyplot(fig)

