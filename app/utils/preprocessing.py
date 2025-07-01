import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data():
    """Load and preprocess Framingham dataset"""
    df = pd.read_csv('data/framingham.csv')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[['glucose', 'heartRate', 'BMI']] = imputer.fit_transform(df[['glucose', 'heartRate', 'BMI']])
    df.dropna(subset=['cigsPerDay', 'totChol', 'BPMeds'], inplace=True)
    
    # Feature engineering
    df['BP_ratio'] = df['sysBP'] / df['diaBP']
    df.drop(['education'], axis=1, inplace=True)  # Low predictive value
    
    # Split data
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'BP_ratio']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save artifacts
    joblib.dump(scaler, 'app/models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test, X.columns