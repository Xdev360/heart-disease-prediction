from flask import render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from .utils.explainability import generate_global_explanations, generate_local_explanation, generate_text_explanation
from .utils.preprocessing import load_and_preprocess_data

def create_routes(app):
    # Load model and artifacts at startup
    try:
        model = joblib.load('app/models/prediction_model.pkl')
        scaler = joblib.load('app/models/scaler.pkl')
        feature_names = pd.read_csv('app/models/feature_names.csv').squeeze()
        # Precompute global explanations
        X_train, _, _, _, _ = load_and_preprocess_data()
        global_shap_img = generate_global_explanations(model, X_train)
    except FileNotFoundError:
        model = None
        scaler = None
        feature_names = None
        global_shap_img = None

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if model is None or scaler is None:
            flash('Model not trained yet. Please train the model first.', 'danger')
            return redirect(url_for('home'))
        try:
            form_data = {
                'male': int(request.form.get('male')),
                'age': float(request.form.get('age')),
                'currentSmoker': int(request.form.get('currentSmoker')),
                'cigsPerDay': float(request.form.get('cigsPerDay')),
                'BPMeds': float(request.form.get('BPMeds')),
                'prevalentStroke': int(request.form.get('prevalentStroke')),
                'prevalentHyp': int(request.form.get('prevalentHyp')),
                'diabetes': int(request.form.get('diabetes')),
                'totChol': float(request.form.get('totChol')),
                'sysBP': float(request.form.get('sysBP')),
                'diaBP': float(request.form.get('diaBP')),
                'BMI': float(request.form.get('BMI')),
                'heartRate': float(request.form.get('heartRate')),
                'glucose': float(request.form.get('glucose'))
            }
            input_df = pd.DataFrame([form_data])
            input_df['BP_ratio'] = input_df['sysBP'] / input_df['diaBP']
            num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'BP_ratio']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            probability = model.predict_proba(input_df)[0][1]
            risk_percentage = round(probability * 100, 2)
            shap_html = generate_local_explanation(model, input_df)
            text_explanation = generate_text_explanation(model, input_df, feature_names=feature_names)
            return render_template(
                'result.html',
                risk=risk_percentage,
                shap_plot=shap_html,
                global_shap_img=global_shap_img,
                text_explanation=text_explanation
            )
        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'danger')
            return redirect(url_for('home'))

    @app.route('/train', methods=['POST'])
    def train_model():
        try:
            from .models.model_training import train_and_evaluate_model
            model, metrics = train_and_evaluate_model()
            flash('Model trained successfully!', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            flash(f'Error training model: {str(e)}', 'danger')
            return redirect(url_for('home'))