# streamlit_app.py
import streamlit as st
import pandas as pd
from PIL import Image
from pycaret.classification import load_model, predict_model

# Setup
st.set_page_config(layout='wide')
st.title("Patient With Angina Prediction")

# Load model
model = load_model('All_Variables_Model_LightGBM')

# Sidebar inputs
st.sidebar.image('ambulance.jpg', use_container_width=True)
st.sidebar.header('Patient Input Features')

# Default input values
default_values = {
    'chest_pain': 0.0, 'age': 51, 'sex': 'Female', 'ethnic': 'White European', 'BMI': 20.2115,
    'smoking_status': 'non-smoker', 'physical_activity': 'high', 'mean_sbp': 116, 'mean_dbp': 79.5,
    'mean_heart_rate': 61, 'hba1c': 38.5, 'random_glucose': 5.995, 'total_cholesterol': 4.47,
    'hdl': 1.492, 'ldl': 2.69, 'triglyceride': 0.504, 'Cholesterol_HDL_Ratio': 2.996, 'fam_chd': 1,
    'chol_lowering': 0, 'has_t1d': 0, 'has_t2d': 0, 'diabetes_status': 'No Diabetes',
    'treated_hypertension': 0, 'corticosteroid_use': 0, 'creatinine': 52, 'blood_urea_nitrogen': 2.36,
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

# with left_col:
#    st.subheader("Feature Importance")
#    st.image('Preprocessed_model(important variables only) 2.png', use_container_width=True)

with right_col:
    st.subheader("Predicted Patients with Angina")
    if st.button('Predict Patient With Angina'):
        pred = predict_model(model, data=input_df)
        label = pred['prediction_label'][0]
        if label == 1:
            st.error("Predicted Patient With Angina: High Risk 🚨")
        else:
            st.success("Predicted Patient With Angina: Low Risk ✅")
    else:
        st.info("Enter inputs on the left and click 'Predict Patient With Angina'")