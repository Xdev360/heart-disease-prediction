import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from app.utils.preprocessing import load_and_preprocess_data
import joblib
import pandas as pd

def train_and_evaluate_model():
    """Train XGBoost model with hyperparameter tuning"""
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Model with initial parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        n_estimators=200,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    
    # Save model and metadata
    joblib.dump(best_model, 'app/models/prediction_model.pkl')
    pd.Series(feature_names).to_csv('app/models/feature_names.csv', index=False)
    
    return best_model, metrics 