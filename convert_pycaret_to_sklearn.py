# convert_pycaret_to_sklearn.py
import pandas as pd
import joblib
import pickle
from pycaret.classification import load_model, finalize_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def extract_sklearn_components():
    """Extract sklearn components from PyCaret model"""
    
    # Load your PyCaret model
    pycaret_model = load_model('All_Variables_Model_LightGBM')
    
    # Get the underlying sklearn model
    sklearn_model = pycaret_model
    
    # Create sample data to understand preprocessing
    sample_data = pd.DataFrame({
        'chest_pain': [0.0], 'age': [51], 'sex': ['Female'], 'ethnic': ['White European'], 
        'BMI': [20.2115], 'smoking_status': ['non-smoker'], 'physical_activity': ['high'], 
        'mean_sbp': [116], 'mean_dbp': [79.5], 'mean_heart_rate': [61], 'hba1c': [38.5], 
        'random_glucose': [5.995], 'total_cholesterol': [4.47], 'hdl': [1.492], 'ldl': [2.69], 
        'triglyceride': [0.504], 'Cholesterol_HDL_Ratio': [2.996], 'fam_chd': [1], 
        'chol_lowering': [0], 'has_t1d': [0], 'has_t2d': [0], 'diabetes_status': ['No Diabetes'],
        'treated_hypertension': [0], 'corticosteroid_use': [0], 'creatinine': [52], 
        'blood_urea_nitrogen': [2.36], 'sodium': [14], 'potassium': [13.6], 'glucose': [5.995], 
        'hemoglobin': [11.93], 'hematocrit': [35.34], 'mean_corpuscular_volume': [91.24], 
        'mean_corpuscular_hemoglobin': [30.79], 'mean_corpuscular_hemoglobin_concentration': [33.75], 
        'white_blood_cell_count': [5.24], 'red_blood_cell_count': [3.873], 'platelet_count': [242.7], 
        'creatine_phosphokinase': [1690], 'ast': [24.6], 'uric_acid': [131.7]
    })
    
    # Save the sklearn model
    joblib.dump(sklearn_model, 'sklearn_model.pkl')
    
    # Create and save label encoders for categorical variables
    label_encoders = {}
    categorical_cols = ['sex', 'ethnic', 'smoking_status', 'physical_activity', 'diabetes_status']
    
    # Define all possible values for each categorical variable
    categorical_values = {
        'sex': ['Female', 'Male'],
        'ethnic': ['White European', 'Black African', 'Black Caribbean', 'Chinese', 'Mixed', 'Other ethnic group', 'South Asian'],
        'smoking_status': ['non-smoker', 'ex-smoker', 'light smoker', 'moderate smoker', 'heavy smoker'],
        'physical_activity': ['low', 'moderate', 'high'],
        'diabetes_status': ['No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes']
    }
    
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(categorical_values[col])
        label_encoders[col] = le
    
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    print("Model conversion completed!")
    print("Files saved:")
    print("- sklearn_model.pkl")
    print("- label_encoders.pkl")
    
    return sklearn_model, label_encoders

if __name__ == "__main__":
    extract_sklearn_components()