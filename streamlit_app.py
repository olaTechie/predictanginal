# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Setup
st.set_page_config(layout='wide')
st.title("Patient With Angina Prediction")

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and any preprocessors"""
    try:
        # Load your trained sklearn model
        model = joblib.load('sklearn_model.pkl')
        
        # Load preprocessors if you have them
        try:
            label_encoders = joblib.load('label_encoders.pkl')
            scaler = joblib.load('scaler.pkl')
        except:
            label_encoders = None
            scaler = None
            
        return model, label_encoders, scaler
    except FileNotFoundError:
        st.error("Model file not found. Please ensure sklearn_model.pkl exists.")
        return None, None, None

def preprocess_input(df, label_encoders=None, scaler=None):
    """Preprocess input data to match training data format"""
    df_processed = df.copy()
    
    # Handle categorical variables
    categorical_cols = ['sex', 'ethnic', 'smoking_status', 'physical_activity', 'diabetes_status']
    
    if label_encoders:
        for col in categorical_cols:
            if col in df_processed.columns and col in label_encoders:
                df_processed[col] = label_encoders[col].transform(df_processed[col])
    else:
        # Manual encoding if no label encoders available
        encoding_maps = {
            'sex': {'Female': 0, 'Male': 1},
            'ethnic': {
                'White European': 0, 'Black African': 1, 'Black Caribbean': 2, 
                'Chinese': 3, 'Mixed': 4, 'Other ethnic group': 5, 'South Asian': 6
            },
            'smoking_status': {
                'non-smoker': 0, 'ex-smoker': 1, 'light smoker': 2, 
                'moderate smoker': 3, 'heavy smoker': 4
            },
            'physical_activity': {'low': 0, 'moderate': 1, 'high': 2},
            'diabetes_status': {'No Diabetes': 0, 'Type 1 Diabetes': 1, 'Type 2 Diabetes': 2}
        }
        
        for col, mapping in encoding_maps.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map(mapping)
    
    # Convert boolean columns to integers
    bool_cols = ['chest_pain', 'fam_chd', 'chol_lowering', 'has_t1d', 'has_t2d', 
                 'treated_hypertension', 'corticosteroid_use']
    for col in bool_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int)
    
    # Scale numerical features if scaler is available
    if scaler:
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    return df_processed

# Load model and preprocessors
model, label_encoders, scaler = load_model_and_preprocessors()

# Sidebar inputs
st.sidebar.image('ambulance.jpg')
# st.sidebar.image('ambulance.jpg', use_container_width=True)
st.sidebar.header('Patient Input Features')

# Default input values
default_values = {
    'chest_pain': False, 'age': 51, 'sex': 'Female', 'ethnic': 'White European', 'BMI': 20.2115,
    'smoking_status': 'non-smoker', 'physical_activity': 'high', 'mean_sbp': 116, 'mean_dbp': 79.5,
    'mean_heart_rate': 61, 'hba1c': 38.5, 'random_glucose': 5.995, 'total_cholesterol': 4.47,
    'hdl': 1.492, 'ldl': 2.69, 'triglyceride': 0.504, 'Cholesterol_HDL_Ratio': 2.996, 'fam_chd': True,
    'chol_lowering': False, 'has_t1d': False, 'has_t2d': False, 'diabetes_status': 'No Diabetes',
    'treated_hypertension': False, 'corticosteroid_use': False, 'creatinine': 52, 'blood_urea_nitrogen': 2.36,
    'sodium': 14, 'potassium': 13.6, 'glucose': 5.995, 'hemoglobin': 11.93, 'hematocrit': 35.34,
    'mean_corpuscular_volume': 91.24, 'mean_corpuscular_hemoglobin': 30.79,
    'mean_corpuscular_hemoglobin_concentration': 33.75, 'white_blood_cell_count': 5.24,
    'red_blood_cell_count': 3.873, 'platelet_count': 242.7, 'creatine_phosphokinase': 1690,
    'ast': 24.6, 'uric_acid': 131.7
}

ethnic_options = ['White European', 'Black African', 'Black Caribbean', 'Chinese', 'Mixed', 'Other ethnic group', 'South Asian']
smoking_options = ['ex-smoker', 'heavy smoker', 'light smoker', 'moderate smoker', 'non-smoker']
activity_options = ['high', 'low', 'moderate']
diabetes_options = ['No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes']

# Input fields
inputs = {}
inputs['age'] = st.sidebar.number_input('Age', value=default_values['age'])
inputs['sex'] = st.sidebar.selectbox('Sex', ['Female', 'Male'], index=0 if default_values['sex'] == 'Female' else 1)
inputs['ethnic'] = st.sidebar.selectbox('Ethnic Group', ethnic_options, index=ethnic_options.index(default_values['ethnic']))
inputs['BMI'] = st.sidebar.number_input('BMI', value=default_values['BMI'])
inputs['smoking_status'] = st.sidebar.selectbox('Smoking Status', smoking_options, index=smoking_options.index(default_values['smoking_status']))
inputs['physical_activity'] = st.sidebar.selectbox('Physical Activity', activity_options, index=activity_options.index(default_values['physical_activity']))
inputs['chest_pain'] = st.sidebar.selectbox('Chest Pain', [True, False], index=int(default_values['chest_pain']))
inputs['mean_sbp'] = st.sidebar.number_input('Mean Systolic BP', value=default_values['mean_sbp'])
inputs['mean_dbp'] = st.sidebar.number_input('Mean Diastolic BP', value=default_values['mean_dbp'])
inputs['mean_heart_rate'] = st.sidebar.number_input('Mean Heart Rate', value=default_values['mean_heart_rate'])
inputs['hba1c'] = st.sidebar.number_input('HbA1c', value=default_values['hba1c'])
inputs['random_glucose'] = st.sidebar.number_input('Random Glucose', value=default_values['random_glucose'])
inputs['glucose'] = st.sidebar.number_input('Glucose', value=default_values['glucose'])
inputs['total_cholesterol'] = st.sidebar.number_input('Total Cholesterol', value=default_values['total_cholesterol'])
inputs['hdl'] = st.sidebar.number_input('HDL', value=default_values['hdl'])
inputs['ldl'] = st.sidebar.number_input('LDL', value=default_values['ldl'])
inputs['triglyceride'] = st.sidebar.number_input('Triglyceride', value=default_values['triglyceride'])
inputs['Cholesterol_HDL_Ratio'] = st.sidebar.number_input('Cholesterol/HDL Ratio', value=default_values['Cholesterol_HDL_Ratio'])
inputs['fam_chd'] = st.sidebar.selectbox('Family History of CHD', [True, False], index=int(default_values['fam_chd']))
inputs['chol_lowering'] = st.sidebar.selectbox('Cholesterol Lowering Meds', [True, False], index=int(default_values['chol_lowering']))
inputs['has_t1d'] = st.sidebar.selectbox('Has Type 1 Diabetes', [True, False], index=int(default_values['has_t1d']))
inputs['has_t2d'] = st.sidebar.selectbox('Has Type 2 Diabetes', [True, False], index=int(default_values['has_t2d']))
inputs['diabetes_status'] = st.sidebar.selectbox('Diabetes Status', diabetes_options, index=diabetes_options.index(default_values['diabetes_status']))
inputs['treated_hypertension'] = st.sidebar.selectbox('Treated Hypertension', [True, False], index=int(default_values['treated_hypertension']))
inputs['corticosteroid_use'] = st.sidebar.selectbox('Corticosteroid Use', [True, False], index=int(default_values['corticosteroid_use']))
inputs['creatinine'] = st.sidebar.number_input('Creatinine', value=default_values['creatinine'])
inputs['blood_urea_nitrogen'] = st.sidebar.number_input('Blood Urea Nitrogen', value=default_values['blood_urea_nitrogen'])
inputs['sodium'] = st.sidebar.number_input('Sodium', value=default_values['sodium'])
inputs['potassium'] = st.sidebar.number_input('Potassium', value=default_values['potassium'])
inputs['hemoglobin'] = st.sidebar.number_input('Hemoglobin', value=default_values['hemoglobin'])
inputs['hematocrit'] = st.sidebar.number_input('Hematocrit', value=default_values['hematocrit'])
inputs['mean_corpuscular_volume'] = st.sidebar.number_input('MCV', value=default_values['mean_corpuscular_volume'])
inputs['mean_corpuscular_hemoglobin'] = st.sidebar.number_input('MCH', value=default_values['mean_corpuscular_hemoglobin'])
inputs['mean_corpuscular_hemoglobin_concentration'] = st.sidebar.number_input('MCHC', value=default_values['mean_corpuscular_hemoglobin_concentration'])
inputs['white_blood_cell_count'] = st.sidebar.number_input('WBC Count', value=default_values['white_blood_cell_count'])
inputs['red_blood_cell_count'] = st.sidebar.number_input('RBC Count', value=default_values['red_blood_cell_count'])
inputs['platelet_count'] = st.sidebar.number_input('Platelet Count', value=default_values['platelet_count'])
inputs['creatine_phosphokinase'] = st.sidebar.number_input('Creatine Phosphokinase', value=default_values['creatine_phosphokinase'])
inputs['ast'] = st.sidebar.number_input('AST', value=default_values['ast'])
inputs['uric_acid'] = st.sidebar.number_input('Uric Acid', value=default_values['uric_acid'])

input_df = pd.DataFrame([inputs])

# Layout
left_col, right_col = st.columns(2)

with right_col:
    st.subheader("Predicted Patients with Angina")
    if st.button('Predict Patient With Angina'):
        if model is not None:
            try:
                # Preprocess input
                processed_input = preprocess_input(input_df, label_encoders, scaler)
                
                # Make prediction
                prediction = model.predict(processed_input)[0]
                prediction_proba = model.predict_proba(processed_input)[0]
                
                # Display results
                if prediction == 1:
                    st.error(f"Predicted Patient With Angina: High Risk 🚨")
                    st.write(f"Risk probability: {prediction_proba[1]:.2%}")
                else:
                    st.success(f"Predicted Patient With Angina: Low Risk ✅")
                    st.write(f"Risk probability: {prediction_proba[1]:.2%}")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.error("Model not loaded. Please check your model file.")
    else:
        st.info("Enter inputs on the left and click 'Predict Patient With Angina'")