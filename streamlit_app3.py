import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Import the predictor class
try:
    from predictor import PyCaretPredictor
except ImportError:
    st.error("❌ Could not import predictor module. Please ensure predictor.py is in the same directory.")
    st.stop()

@st.cache_resource
def load_predictor():
    """Load the PyCaret predictor"""
    try:
        # Check if the predictor file exists
        if os.path.exists('pycaret_predictor.pkl'):
            predictor = joblib.load('pycaret_predictor.pkl')
            st.success("✅ Prediction model loaded successfully!")
            return predictor
        else:
            st.error("❌ pycaret_predictor.pkl not found. Please run the conversion script first.")
            st.info("📋 Steps to fix:")
            st.info("1. Run convert_pycaret_final.py locally")
            st.info("2. Copy predictor.py to your app directory")
            st.info("3. Copy pycaret_predictor.pkl to your app directory")
            st.info("4. Redeploy your app")
            return None
            
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        st.error("Make sure both predictor.py and pycaret_predictor.pkl are in your app directory")
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

def display_prediction_results(prediction, probabilities):
    """Display prediction results in a nice format"""
    
    st.header("🎯 Prediction Results")
    
    # Main result
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK**")
            st.error("Likely to have anginal chest pain")
        else:
            st.success("✅ **LOW RISK**")
            st.success("Unlikely to have anginal chest pain")
    
    with col2:
        if probabilities and len(probabilities) >= 2:
            prob_positive = probabilities[1]
            confidence = max(probabilities)
            
            st.metric("Risk Probability", f"{prob_positive:.1%}")
            st.metric("Confidence", f"{confidence:.1%}")
        else:
            st.metric("Prediction", prediction)
    
    # Detailed breakdown
    if probabilities and len(probabilities) >= 2:
        st.subheader("📊 Risk Assessment")
        
        # Create probability dataframe
        prob_df = pd.DataFrame({
            'Outcome': ['No Anginal Pain', 'Anginal Pain'],
            'Probability': [f"{probabilities[0]:.1%}", f"{probabilities[1]:.1%}"]
        })
        
        # Display as metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("No Anginal Pain", f"{probabilities[0]:.1%}")
        with col2:
            st.metric("Anginal Pain", f"{probabilities[1]:.1%}")
        
        # Risk interpretation
        risk_level = probabilities[1]
        
        st.subheader("🔍 Clinical Interpretation")
        if risk_level < 0.3:
            st.success("🟢 **Low Risk**")
            st.success("The patient has a low probability of experiencing anginal chest pain. Continue routine monitoring.")
        elif risk_level < 0.7:
            st.warning("🟡 **Moderate Risk**")
            st.warning("The patient has a moderate probability of experiencing anginal chest pain. Consider further evaluation and monitoring.")
        else:
            st.error("🔴 **High Risk**")
            st.error("The patient has a high probability of experiencing anginal chest pain. Immediate medical attention and comprehensive cardiac evaluation recommended.")
        
        # Progress bar for risk visualization
        st.subheader("🎯 Risk Visualization")
        st.progress(risk_level)
        
        # Additional recommendations
        st.subheader("📋 Recommendations")
        if risk_level < 0.3:
            st.info("• Continue with standard preventive care\n• Regular exercise and healthy diet\n• Routine follow-up appointments")
        elif risk_level < 0.7:
            st.info("• Schedule follow-up within 1-2 weeks\n• Consider stress testing or cardiac imaging\n• Review and optimize risk factors")
        else:
            st.info("• **Immediate medical evaluation required**\n• Emergency cardiac assessment\n• Consider hospital admission if symptoms present")

# App Configuration
st.set_page_config(
    page_title="Anginal Chest Pain Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the predictor
predictor = load_predictor()

# Sidebar
st.sidebar.image('ambulance.jpg', width=200)
st.sidebar.title("Anginal Chest Pain Predictor")
st.sidebar.write("This app predicts the likelihood of anginal chest pain based on patient data.")

# Add model information in sidebar
if predictor is not None:
    st.sidebar.success("✅ Model Status: Ready")
    st.sidebar.info("Model: LightGBM Classifier")
    st.sidebar.info("Features: 40 clinical variables")
else:
    st.sidebar.error("❌ Model Status: Not Available")

# Main app header
st.title("🏥 Anginal Chest Pain Prediction System")
st.markdown("---")

# Check if predictor is loaded
if predictor is None:
    st.error("❌ Prediction system not available. Please ensure the model files are properly deployed.")
    st.stop()

# Create input form
input_data = create_input_form()

# Prediction section
st.markdown("---")
st.header("🔍 Make Prediction")

# Prediction button
if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
    if input_data is not None:
        try:
            with st.spinner("Analyzing patient data..."):
                # Make prediction
                prediction, probabilities = predictor.predict(input_data)
                
                if prediction is not None:
                    # Display results
                    display_prediction_results(prediction, probabilities)
                else:
                    st.error("❌ Prediction failed. Please check your input values and try again.")
                    
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Please verify all input values are within valid ranges.")
            
            # Show debug info in expander
            with st.expander("Debug Information"):
                st.write("Error details:", str(e))
                st.write("Input data shape:", input_data.shape)
                st.write("Input data columns:", list(input_data.columns))

# Footer
st.markdown("---")
st.markdown("### ⚠️ Important Disclaimer")
st.warning("""
**This tool is for educational and research purposes only.**
- Results should not replace professional medical diagnosis
- Always consult with qualified healthcare providers
- This model is based on historical data and may not account for all clinical factors
- Emergency situations require immediate medical attention regardless of prediction results
""")

# Additional information
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    **Model Details:**
    - Algorithm: LightGBM Classifier
    - Features: 40 clinical variables including demographics, vital signs, and laboratory results
    - Training: Based on clinical dataset with anginal chest pain outcomes
    - Validation: Cross-validated performance metrics
    
    **Input Requirements:**
    - All fields should be filled with accurate patient data
    - Laboratory values should be in standard units
    - Missing or invalid values may affect prediction accuracy
    
    **Interpretation:**
    - Low Risk (<30%): Routine monitoring recommended
    - Moderate Risk (30-70%): Further evaluation suggested
    - High Risk (>70%): Immediate medical attention required
    
    **Limitations:**
    - Model performance depends on data quality and completeness
    - Not validated for all patient populations
    - Clinical judgment should always supersede algorithmic predictions
    - Regular model updates and validation are essential
    """)

# Technical details for healthcare providers
with st.expander("🔬 Technical Information"):
    st.markdown("""
    **Model Architecture:**
    - Gradient Boosting Decision Trees (LightGBM)
    - Automated feature engineering and selection
    - Cross-validation with stratified sampling
    - Hyperparameter optimization
    
    **Feature Categories:**
    - Demographics: Age, sex, ethnicity, BMI
    - Lifestyle: Smoking status, physical activity
    - Vital Signs: Blood pressure, heart rate
    - Laboratory: Lipid panel, glucose, kidney function, blood counts
    - Medical History: Family history, diabetes, hypertension, medications
    
    **Performance Metrics:**
    - Accuracy, Precision, Recall, F1-Score
    - Area Under ROC Curve (AUC)
    - Calibration plots and reliability diagrams
    - Feature importance analysis
    
    **Data Sources:**
    - Clinical electronic health records
    - Standardized laboratory reference ranges
    - Validated clinical outcome measures
    """)