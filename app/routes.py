from flask import render_template, request, redirect, url_for, session, flash
import joblib
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from .utils.explainability import generate_global_explanations, generate_local_explanation, generate_text_explanation
from .utils.preprocessing import load_and_preprocess_data

def create_routes(app):
    app.secret_key = 'your_secret_key'  # Needed for session
    
    # Initialize history storage
    HISTORY_FILE = 'prediction_history.json'
    
    def load_history():
        """Load prediction history from JSON file"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(history_data):
        """Save prediction history to JSON file"""
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_to_history(prediction_data):
        """Add a new prediction to history"""
        history = load_history()
        history.append(prediction_data)
        # Keep only last 10 predictions
        if len(history) > 10:
            history = history[-10:]
        save_history(history)
    
    try:
        model = joblib.load('app/models/prediction_model.pkl')
        scaler = joblib.load('app/models/scaler.pkl')
        feature_names = pd.read_csv('app/models/feature_names.csv').squeeze()
        X_train, _, _, _, _ = load_and_preprocess_data()
        global_shap_img = generate_global_explanations(model, X_train)
    except FileNotFoundError:
        model = None
        scaler = None
        feature_names = None
        global_shap_img = None

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/start', methods=['GET', 'POST'])
    def start():
        if request.method == 'POST':
            # Validate age
            try:
                age = int(request.form['age'])
                if age < 30:
                    flash('Age must be 30 or older for accurate heart disease risk assessment.', 'error')
                    return render_template('start.html')
            except (ValueError, KeyError):
                flash('Please enter a valid age.', 'error')
                return render_template('start.html')
            
            session['first_name'] = request.form['first_name']
            session['last_name'] = request.form['last_name']
            session['gender'] = request.form['gender']
            session['age'] = request.form['age']
            
            return redirect(url_for('medical'))
        return render_template('start.html')

    @app.route('/medical', methods=['GET', 'POST'])
    def medical():
        if request.method == 'POST':
            session['bp_meds'] = request.form['bp_meds']
            session['smoke_history'] = request.form['smoke_history']
            session['hypertension'] = request.form['hypertension']
            session['diabetes'] = request.form['diabetes']
            
            return redirect(url_for('lifestyle'))
        return render_template('medical.html')

    @app.route('/lifestyle', methods=['GET', 'POST'])
    def lifestyle():
        if request.method == 'POST':
            print('LIFESTYLE FORM DATA:', request.form)
            session['current_smoker'] = request.form['current_smoker']
            session['cigs_per_day'] = request.form['cigs_per_day']
            
            return redirect(url_for('clinical'))
        return render_template('lifestyle.html')

    @app.route('/clinical', methods=['GET', 'POST'])
    def clinical():
        if request.method == 'POST':
            print('CLINICAL FORM DATA:', request.form)
            session['tot_chol'] = request.form['tot_chol']
            session['heart_rate'] = request.form['heart_rate']
            session['sys_bp'] = request.form['sys_bp']
            session['bmi'] = request.form['bmi']
            session['dia_bp'] = request.form['dia_bp']
            session['glucose'] = request.form['glucose']
            
            # Add a 5-second delay to show the preloader for AI prediction
            time.sleep(5)
            
            return redirect(url_for('result'))
        return render_template('clinical.html')

    @app.route('/result')
    def result():
        # Collect all data from session
        data = {
            'first_name': session.get('first_name'),
            'last_name': session.get('last_name'),
            'gender': session.get('gender'),
            'age': session.get('age'),
            'bp_meds': session.get('bp_meds'),
            'smoke_history': session.get('smoke_history'),
            'hypertension': session.get('hypertension'),
            'diabetes': session.get('diabetes'),
            'current_smoker': session.get('current_smoker'),
            'cigs_per_day': session.get('cigs_per_day'),
            'tot_chol': session.get('tot_chol'),
            'heart_rate': session.get('heart_rate'),
            'sys_bp': session.get('sys_bp'),
            'bmi': session.get('bmi'),
            'dia_bp': session.get('dia_bp'),
            'glucose': session.get('glucose'),
        }

        # Preprocess user input for prediction
        def parse_range_value(value):
            if value is None or value == '':
                return 0
            if '-' in value:
                start, end = value.split('-')
                return (float(start) + float(end)) / 2
            if '+' in value:
                return float(value.replace('+', ''))
            try:
                return float(value)
            except ValueError:
                return 0

        def preprocess_user_input(data, scaler, feature_names):
            # Map session data to DataFrame with correct types and columns
            input_dict = {
                'age': float(data['age']),
                'cigsPerDay': parse_range_value(data['cigs_per_day']),
                'totChol': parse_range_value(data['tot_chol']),
                'sysBP': parse_range_value(data['sys_bp']),
                'diaBP': parse_range_value(data['dia_bp']),
                'BMI': parse_range_value(data['bmi']),
                'heartRate': parse_range_value(data['heart_rate']),
                'glucose': parse_range_value(data['glucose']),
                'male': 1 if data['gender'] == 'male' else 0,
                'BPMeds': 1 if data['bp_meds'] == 'yes' else 0,
                'prevalentStroke': 1 if data['smoke_history'] == 'yes' else 0,
                'prevalentHyp': 1 if data['hypertension'] == 'yes' else 0,
                'diabetes': 1 if data['diabetes'] == 'yes' else 0,
                'currentSmoker': 1 if data['current_smoker'] == 'yes' else 0,
            }
            # Feature engineering
            input_dict['BP_ratio'] = input_dict['sysBP'] / input_dict['diaBP'] if input_dict['diaBP'] != 0 else 0
            # Ensure all features in the right order
            input_df = pd.DataFrame([input_dict])
            # Reorder columns to match training
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            # Scale numerical features
            num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'BP_ratio']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            return input_df

        # Only proceed if model and scaler are loaded
        if model is not None and scaler is not None and feature_names is not None:
            # Preprocess input
            input_df = preprocess_user_input(data, scaler, feature_names)
            # Predict risk (probability)
            risk_proba = model.predict_proba(input_df)[0][1]
            risk_percentage = round(risk_proba * 100, 1)
            # SHAP explanations
            summary = generate_text_explanation(model, input_df, feature_names=feature_names)
            explanation = generate_local_explanation(model, input_df)
            # Global feature importance (already loaded as global_shap_img)
            feature_importance = global_shap_img
        else:
            risk_percentage = 0.0
            summary = "Model not loaded."
            explanation = "Model not loaded."
            feature_importance = "Model not loaded."

        # Save prediction to history
        if data['first_name'] and data['last_name']:
            prediction_record = {
                'id': len(load_history()) + 1,
                'timestamp': datetime.now().isoformat(),
                'patient_name': f"{data['first_name']} {data['last_name']}",
                'risk_percentage': risk_percentage,
                'risk_level': 'Medium' if risk_percentage < 30 else 'High' if risk_percentage > 50 else 'Low',
                'data': data
            }
            add_to_history(prediction_record)

            return render_template(
                'result.html',
                risk=risk_percentage,
                summary=summary,
                explanation=explanation,
                feature_importance=feature_importance,
                patient_name=f"{data['first_name']} {data['last_name']}"
            )

    @app.route('/history')
    def get_history():
        """API endpoint to get prediction history"""
        history = load_history()
        return json.dumps(history)
    
    @app.route('/history/<int:history_id>', methods=['DELETE'])
    def delete_history(history_id):
        """API endpoint to delete a specific prediction from history"""
        try:
            history = load_history()
            # Find and remove the item with the specified ID
            history = [item for item in history if item.get('id') != history_id]
            save_history(history)
            return json.dumps({'success': True, 'message': 'History item deleted successfully'})
        except Exception as e:
            return json.dumps({'success': False, 'message': f'Error deleting history item: {str(e)}'}), 500

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