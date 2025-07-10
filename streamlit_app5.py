import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

@st.cache_resource
def load_pycaret_model():
    """Load the PyCaret model"""
    try:
        # Load your PyCaret model (without .pkl extension)
        model = load_model('All_Variables_Model_LightGBM')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_input_form():
    """Create input form for user data"""
    
    st.header("🩺 Anginal Chest Pain Prediction")
    st.write("Please enter the patient information below:")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        ethnic = st.selectbox("Ethnicity", [
            "White European", "South Asian", "Black African", 
            "Black Caribbean", "Chinese", "Mixed", "Other ethnic group"
        ])
        BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        
        st.subheader("Lifestyle")
        smoking_status = st.selectbox("Smoking Status", [
            "non-smoker", "ex-smoker", "light smoker", "moderate smoker", "heavy smoker"
        ])
        physical_activity = st.selectbox("Physical Activity", ["low", "moderate", "high"])
        
        st.subheader("Chest Pain")
        chest_pain = st.number_input("Chest Pain Score", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    with col2:
        st.subheader("Vital Signs")
        mean_sbp = st.number_input("Mean SBP", min_value=80, max_value=200, value=120)
        mean_dbp = st.number_input("Mean DBP", min_value=50, max_value=120, value=80)
        mean_heart_rate = st.number_input("Mean Heart Rate", min_value=40, max_value=150, value=70)
        
        st.subheader("Blood Tests - Metabolic")
        hba1c = st.number_input("HbA1c", min_value=20.0, max_value=100.0, value=40.0, step=0.1)
        random_glucose = st.number_input("Random Glucose", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        glucose = st.number_input("Glucose", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        
        st.subheader("Lipid Panel")
        total_cholesterol = st.number_input("Total Cholesterol", min_value=2.0, max_value=10.0, value=5.0, step=0.1)
        hdl = st.number_input("HDL", min_value=0.5, max_value=3.0, value=1.3, step=0.1)
        ldl = st.number_input("LDL", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        triglyceride = st.number_input("Triglyceride", min_value=0.3, max_value=5.0, value=1.0, step=0.1)
    
    # Additional fields in expandable sections
    with st.expander("Medical History"):
        col3, col4 = st.columns(2)
        with col3:
            fam_chd = st.selectbox("Family CHD History", [0, 1])
            diabetes_status = st.selectbox("Diabetes Status", 
                                         ["No Diabetes", "Type 1 Diabetes", "Type 2 Diabetes"])
            has_t1d = st.selectbox("Has Type 1 Diabetes", [0, 1])
            has_t2d = st.selectbox("Has Type 2 Diabetes", [0, 1])
        with col4:
            chol_lowering = st.selectbox("Cholesterol Lowering Medication", [0, 1])
            treated_hypertension = st.selectbox("Treated Hypertension", [0, 1])
            corticosteroid_use = st.selectbox("Corticosteroid Use", [0, 1])
    
    with st.expander("Additional Blood Tests"):
        col5, col6 = st.columns(2)
        with col5:
            creatinine = st.number_input("Creatinine", min_value=30.0, max_value=300.0, value=80.0, step=1.0)
            blood_urea_nitrogen = st.number_input("Blood Urea Nitrogen", min_value=1.0, max_value=20.0, value=4.0, step=0.1)
            sodium = st.number_input("Sodium", min_value=10.0, max_value=20.0, value=14.0, step=0.1)
            potassium = st.number_input("Potassium", min_value=10.0, max_value=20.0, value=14.0, step=0.1)
            uric_acid = st.number_input("Uric Acid", min_value=100.0, max_value=600.0, value=300.0, step=1.0)
        with col6:
            hemoglobin = st.number_input("Hemoglobin", min_value=8.0, max_value=18.0, value=12.0, step=0.1)
            hematocrit = st.number_input("Hematocrit", min_value=25.0, max_value=50.0, value=36.0, step=0.1)
            white_blood_cell_count = st.number_input("WBC Count", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
            red_blood_cell_count = st.number_input("RBC Count", min_value=3.0, max_value=6.0, value=4.5, step=0.1)
            platelet_count = st.number_input("Platelet Count", min_value=100.0, max_value=500.0, value=250.0, step=1.0)
    
    with st.expander("Additional Measurements"):
        col7, col8 = st.columns(2)
        with col7:
            mean_corpuscular_volume = st.number_input("MCV", min_value=70.0, max_value=110.0, value=90.0, step=0.1)
            mean_corpuscular_hemoglobin = st.number_input("MCH", min_value=25.0, max_value=35.0, value=30.0, step=0.1)
            mean_corpuscular_hemoglobin_concentration = st.number_input("MCHC", min_value=30.0, max_value=38.0, value=33.0, step=0.1)
        with col8:
            creatine_phosphokinase = st.number_input("CPK", min_value=50.0, max_value=3000.0, value=150.0, step=1.0)
            ast = st.number_input("AST", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    
    # Calculate derived features
    Cholesterol_HDL_Ratio = total_cholesterol / hdl if hdl > 0 else 0
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'chest_pain': [chest_pain],
        'age': [age],
        'sex': [sex],
        'ethnic': [ethnic],
        'BMI': [BMI],
        'smoking_status': [smoking_status],
        'physical_activity': [physical_activity],
        'mean_sbp': [mean_sbp],
        'mean_dbp': [mean_dbp],
        'mean_heart_rate': [mean_heart_rate],
        'hba1c': [hba1c],
        'random_glucose': [random_glucose],
        'total_cholesterol': [total_cholesterol],
        'hdl': [hdl],
        'ldl': [ldl],
        'triglyceride': [triglyceride],
        'Cholesterol_HDL_Ratio': [Cholesterol_HDL_Ratio],
        'fam_chd': [fam_chd],
        'chol_lowering': [chol_lowering],
        'has_t1d': [has_t1d],
        'has_t2d': [has_t2d],
        'diabetes_status': [diabetes_status],
        'treated_hypertension': [treated_hypertension],
        'corticosteroid_use': [corticosteroid_use],
        'creatinine': [creatinine],
        'blood_urea_nitrogen': [blood_urea_nitrogen],
        'sodium': [sodium],
        'potassium': [potassium],
        'glucose': [glucose],
        'hemoglobin': [hemoglobin],
        'hematocrit': [hematocrit],
        'mean_corpuscular_volume': [mean_corpuscular_volume],
        'mean_corpuscular_hemoglobin': [mean_corpuscular_hemoglobin],
        'mean_corpuscular_hemoglobin_concentration': [mean_corpuscular_hemoglobin_concentration],
        'white_blood_cell_count': [white_blood_cell_count],
        'red_blood_cell_count': [red_blood_cell_count],
        'platelet_count': [platelet_count],
        'creatine_phosphokinase': [creatine_phosphokinase],
        'ast': [ast],
        'uric_acid': [uric_acid]
    })
    
    return input_data

# Load PyCaret model
model = load_pycaret_model()

if model is None:
    st.error("❌ Could not load the PyCaret model. Please check if the model file exists.")
    st.stop()

# Sidebar
try:
    st.sidebar.image('ambulance.jpg', width=200)
except:
    st.sidebar.write("🚑")  # Fallback if image not found

st.sidebar.title("Anginal Chest Pain Predictor")
st.sidebar.write("This app predicts the likelihood of anginal chest pain based on patient data using PyCaret.")
st.sidebar.write("---")
st.sidebar.info("📋 **Model Information**\n\n"
                "- Algorithm: LightGBM\n"
                "- Framework: PyCaret\n"
                "- Features: 35+ clinical variables")

# Main app
st.title("🏥 Anginal Chest Pain Prediction System")
st.markdown("*Powered by PyCaret & LightGBM*")

# Create input form
input_data = create_input_form()

# Prediction button
if st.button("🔍 Predict", type="primary"):
    if input_data is not None:
        try:
            with st.spinner("Making prediction..."):
                # Use PyCaret's predict_model function
                predictions = predict_model(model, data=input_data)
                
                # Extract prediction and probability
                # PyCaret typically adds columns like 'prediction_label' and 'prediction_score'
                prediction_label = predictions['prediction_label'].iloc[0]
                prediction_score = predictions['prediction_score'].iloc[0]
                
                # Display results
                st.header("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction_label == 1:
                        st.error("⚠️ HIGH RISK: Likely to have anginal chest pain")
                    else:
                        st.success("✅ LOW RISK: Unlikely to have anginal chest pain")
                
                with col2:
                    st.metric("Prediction Probability", f"{prediction_score:.1%}")
                    confidence = max(prediction_score, 1 - prediction_score)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Show detailed results
                st.subheader("📊 Detailed Analysis")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Outcome': ['No Anginal Pain', 'Anginal Pain'],
                    'Probability': [f"{1-prediction_score:.1%}", f"{prediction_score:.1%}"]
                })
                st.dataframe(results_df, use_container_width=True)
                
                # Risk interpretation
                if prediction_score < 0.3:
                    st.info("🟢 **Low Risk**: The patient has a low probability of experiencing anginal chest pain.")
                elif prediction_score < 0.7:
                    st.warning("🟡 **Moderate Risk**: The patient has a moderate probability of experiencing anginal chest pain. Consider further evaluation.")
                else:
                    st.error("🔴 **High Risk**: The patient has a high probability of experiencing anginal chest pain. Immediate medical attention recommended.")
                
                # Show input summary in expander
                with st.expander("📋 Input Summary"):
                    st.write("**Patient Data Used for Prediction:**")
                    
                    # Display key inputs in a nice format
                    summary_data = {
                        'Demographics': {
                            'Age': f"{input_data['age'].iloc[0]} years",
                            'Sex': input_data['sex'].iloc[0],
                            'Ethnicity': input_data['ethnic'].iloc[0],
                            'BMI': f"{input_data['BMI'].iloc[0]:.1f}"
                        },
                        'Vital Signs': {
                            'Blood Pressure': f"{input_data['mean_sbp'].iloc[0]}/{input_data['mean_dbp'].iloc[0]} mmHg",
                            'Heart Rate': f"{input_data['mean_heart_rate'].iloc[0]} bpm",
                            'Chest Pain Score': f"{input_data['chest_pain'].iloc[0]}"
                        },
                        'Key Lab Values': {
                            'HbA1c': f"{input_data['hba1c'].iloc[0]}",
                            'Total Cholesterol': f"{input_data['total_cholesterol'].iloc[0]:.1f}",
                            'HDL': f"{input_data['hdl'].iloc[0]:.1f}",
                            'Cholesterol/HDL Ratio': f"{input_data['Cholesterol_HDL_Ratio'].iloc[0]:.1f}"
                        }
                    }
                    
                    for category, values in summary_data.items():
                        st.write(f"**{category}:**")
                        for key, value in values.items():
                            st.write(f"- {key}: {value}")
                        st.write("")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.error("Please check that all input values are valid.")
            st.write("**Debug Info:**")
            st.write(f"Input data shape: {input_data.shape}")
            st.write(f"Input data columns: {list(input_data.columns)}")

# Information section
with st.expander("ℹ️ About This Model"):
    st.write("""
    **Model Details:**
    - **Algorithm**: LightGBM (Light Gradient Boosting Machine)
    - **Framework**: PyCaret
    - **Input Features**: 35+ clinical variables including demographics, vital signs, laboratory values, and medical history
    - **Output**: Binary classification (Anginal vs Non-anginal chest pain)
    
    **Clinical Variables Used:**
    - Demographics: Age, sex, ethnicity, BMI
    - Lifestyle: Smoking status, physical activity
    - Vital signs: Blood pressure, heart rate
    - Laboratory tests: Lipid panel, glucose, HbA1c, kidney function, liver function, complete blood count
    - Medical history: Family history, diabetes, hypertension, medications
    
    **Model Performance:**
    - The model has been trained on clinical data to distinguish between anginal and non-anginal chest pain
    - Results should be interpreted by qualified healthcare professionals
    """)

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This tool is for educational and research purposes only and should not replace professional medical diagnosis or clinical judgment.")
st.markdown("🔬 **Model**: LightGBM via PyCaret | 📊 **Framework**: Streamlit")