import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import joblib
import pandas as pd

def generate_global_explanations(model, X_train):
    """Generate global SHAP explanations"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_local_explanation(model, input_data):
    """Generate local SHAP explanation for a single prediction"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Force plot
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values,
        input_data,
        matplotlib=False,
        show=False
    )
    
    # Convert to HTML
    shap_html = f"<head>{shap.getjs()}</head>{force_plot.html()}"
    return shap_html

def generate_text_explanation(model, input_data, feature_names=None, top_n=3):
    """Generate a text explanation for a single prediction using SHAP values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    # If input_data is a DataFrame, get the first row
    if hasattr(input_data, 'iloc'):
        shap_row = shap_values[0] if hasattr(shap_values, '__len__') and len(shap_values.shape) > 1 else shap_values
        shap_row = shap_row[0] if hasattr(shap_row, '__len__') and len(shap_row.shape) > 1 else shap_row
    else:
        shap_row = shap_values
    # Get feature names
    if feature_names is None:
        if hasattr(input_data, 'columns'):
            feature_names = input_data.columns
        else:
            feature_names = [f'feature_{i}' for i in range(len(shap_row))]
    # Pair feature names with SHAP values
    feature_impact = list(zip(feature_names, shap_row))
    # Sort by absolute impact
    feature_impact_sorted = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
    # Top features increasing risk
    top_increase = [f for f in feature_impact_sorted if f[1] > 0][:top_n]
    # Top features decreasing risk
    top_decrease = [f for f in feature_impact_sorted if f[1] < 0][:top_n]
    def fmt(feats):
        return ', '.join([f"{name.replace('_', ' ')}" for name, _ in feats]) if feats else 'none'
    text = ""
    if top_increase:
        text += f"The main factors increasing your risk are: {fmt(top_increase)}. "
    if top_decrease:
        text += f"The main factors decreasing your risk are: {fmt(top_decrease)}. "
    if not text:
        text = "No significant factors identified for this prediction."
    return text.strip()