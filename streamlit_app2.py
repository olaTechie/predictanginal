# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration with custom theme
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .high-risk {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
        color: white;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #56AB2F, #A8E6CF);
        color: white;
    }
    
    .info-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .feature-importance {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stSelectbox label, .stNumberInput label {
        font-weight: 600;
        color: #2C3E50;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and any preprocessors"""
    try:
        model = joblib.load('sklearn_model.pkl')
        try:
            label_encoders = joblib.load('label_encoders.pkl')
            scaler = joblib.load('scaler.pkl')
        except:
            label_encoders = None
            scaler = None
        return model, label_encoders, scaler
    except FileNotFoundError:
        st.error("🚨 Model file not found. Please ensure sklearn_model.pkl exists.")
        return None, None, None

def preprocess_input(df, label_encoders=None, scaler=None):
    """Preprocess input data to match training data format"""
    df_processed = df.copy()
    
    categorical_cols = ['sex', 'ethnic', 'smoking_status', 'physical_activity', 'diabetes_status']
    
    if label_encoders:
        for col in categorical_cols:
            if col in df_processed.columns and col in label_encoders:
                df_processed[col] = label_encoders[col].transform(df_processed[col])
    else:
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
    
    bool_cols = ['chest_pain', 'fam_chd', 'chol_lowering', 'has_t1d', 'has_t2d', 
                 'treated_hypertension', 'corticosteroid_use']
    for col in bool_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int)
    
    if scaler:
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    return df_processed

def create_gauge_chart(probability):
    """Create a beautiful gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Level (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        height=300
    )
    return fig

def create_feature_radar(inputs):
    """Create a radar chart showing key health metrics"""
    # Normalize some key metrics for visualization
    metrics = {
        'Age': min(inputs['age'] / 80, 1),
        'BMI': min(inputs['BMI'] / 40, 1),
        'Blood Pressure': min(inputs['mean_sbp'] / 160, 1),
        'Cholesterol': min(inputs['total_cholesterol'] / 8, 1),
        'Heart Rate': min(inputs['mean_heart_rate'] / 120, 1),
        'Glucose': min(inputs['glucose'] / 15, 1)
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name='Patient Profile',
        line_color='rgb(255, 99, 132)',
        fillcolor='rgba(255, 99, 132, 0.6)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Health Metrics Overview",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Header
st.markdown('<h1 class="main-header">🫀 CardioPredict AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Angina Risk Assessment System</p>', unsafe_allow_html=True)

# Load model
model, label_encoders, scaler = load_model_and_preprocessors()

# Sidebar with beautiful styling
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2>📋 Patient Information</h2></div>', unsafe_allow_html=True)
    
    # Patient image or logo
    try:
        st.image('ambulance.jpg', use_container_width=True, caption="Medical Assessment")
    except:
        st.markdown("🏥", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Default values
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
    
    # Organize inputs into tabs for better UX
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🩺 Vitals", "🧪 Labs"])
    
    inputs = {}
    
    with tab1:
        st.subheader("Basic Information")
        inputs['age'] = st.number_input('Age (years)', min_value=18, max_value=120, value=default_values['age'])
        inputs['sex'] = st.selectbox('Sex', ['Female', 'Male'], index=0 if default_values['sex'] == 'Female' else 1)
        inputs['ethnic'] = st.selectbox('Ethnic Group', 
                                      ['White European', 'Black African', 'Black Caribbean', 'Chinese', 'Mixed', 'Other ethnic group', 'South Asian'],
                                      index=0)
        inputs['BMI'] = st.number_input('BMI (kg/m²)', min_value=10.0, max_value=60.0, value=default_values['BMI'], format="%.2f")
        inputs['smoking_status'] = st.selectbox('Smoking Status', 
                                              ['non-smoker', 'ex-smoker', 'light smoker', 'moderate smoker', 'heavy smoker'],
                                              index=0)
        inputs['physical_activity'] = st.selectbox('Physical Activity Level', ['low', 'moderate', 'high'], index=2)
    
    with tab2:
        st.subheader("Vital Signs & Symptoms")
        inputs['chest_pain'] = st.selectbox('Chest Pain', [False, True], index=0)
        inputs['mean_sbp'] = st.number_input('Systolic BP (mmHg)', min_value=70, max_value=250, value=default_values['mean_sbp'])
        inputs['mean_dbp'] = st.number_input('Diastolic BP (mmHg)', min_value=40, max_value=150, value=int(default_values['mean_dbp']))
        inputs['mean_heart_rate'] = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200, value=default_values['mean_heart_rate'])
        
        st.subheader("Medical History")
        inputs['fam_chd'] = st.selectbox('Family History of CHD', [False, True], index=1)
        inputs['diabetes_status'] = st.selectbox('Diabetes Status', ['No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes'], index=0)
        inputs['treated_hypertension'] = st.selectbox('Treated Hypertension', [False, True], index=0)
        inputs['chol_lowering'] = st.selectbox('Cholesterol Medication', [False, True], index=0)
        inputs['corticosteroid_use'] = st.selectbox('Corticosteroid Use', [False, True], index=0)
    
    with tab3:
        st.subheader("Blood Chemistry")
        col1, col2 = st.columns(2)
        
        with col1:
            inputs['total_cholesterol'] = st.number_input('Total Cholesterol', value=default_values['total_cholesterol'], format="%.2f")
            inputs['hdl'] = st.number_input('HDL', value=default_values['hdl'], format="%.2f")
            inputs['ldl'] = st.number_input('LDL', value=default_values['ldl'], format="%.2f")
            inputs['glucose'] = st.number_input('Glucose', value=default_values['glucose'], format="%.2f")
            inputs['hba1c'] = st.number_input('HbA1c', value=default_values['hba1c'], format="%.1f")
        
        with col2:
            inputs['triglyceride'] = st.number_input('Triglycerides', value=default_values['triglyceride'], format="%.3f")
            inputs['creatinine'] = st.number_input('Creatinine', value=default_values['creatinine'])
            inputs['hemoglobin'] = st.number_input('Hemoglobin', value=default_values['hemoglobin'], format="%.2f")
            inputs['platelet_count'] = st.number_input('Platelet Count', value=default_values['platelet_count'], format="%.1f")
            inputs['uric_acid'] = st.number_input('Uric Acid', value=default_values['uric_acid'], format="%.1f")
        
        # Add remaining inputs with default values
        remaining_inputs = {k: v for k, v in default_values.items() if k not in inputs}
        inputs.update(remaining_inputs)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Health metrics visualization
    st.subheader("📊 Health Metrics Overview")
    radar_chart = create_feature_radar(inputs)
    st.plotly_chart(radar_chart, use_container_width=True)

with col2:
    # Current stats
    st.subheader("📈 Current Stats")
    
    # Risk factors count
    risk_factors = sum([
        inputs['chest_pain'],
        inputs['age'] > 65,
        inputs['BMI'] > 30,
        inputs['mean_sbp'] > 140,
        inputs['smoking_status'] != 'non-smoker',
        inputs['fam_chd'],
        inputs['diabetes_status'] != 'No Diabetes'
    ])
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>🚨 Risk Factors</h3>
        <h1>{risk_factors}/7</h1>
    </div>
    """, unsafe_allow_html=True)

# Prediction section
st.markdown("---")
st.subheader("🔮 AI Prediction Results")

input_df = pd.DataFrame([inputs])

if st.button('🧠 Analyze Patient Risk', use_container_width=True, type="primary"):
    if model is not None:
        try:
            # Show loading animation
            with st.spinner('🔄 Analyzing patient data...'):
                time.sleep(2)  # Simulate processing time
                
                # Preprocess and predict
                processed_input = preprocess_input(input_df, label_encoders, scaler)
                prediction = model.predict(processed_input)[0]
                prediction_proba = model.predict_proba(processed_input)[0]
                
                # Results layout
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-card high-risk">
                            <h2>⚠️ HIGH RISK</h2>
                            <h3>Angina Risk Detected</h3>
                            <p style="font-size: 1.2rem;">Risk Probability: {prediction_proba[1]:.1%}</p>
                            <p>Immediate medical consultation recommended</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card low-risk">
                            <h2>✅ LOW RISK</h2>
                            <h3>No Immediate Angina Risk</h3>
                            <p style="font-size: 1.2rem;">Risk Probability: {prediction_proba[1]:.1%}</p>
                            <p>Continue regular health monitoring</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Risk gauge
                st.subheader("📊 Risk Assessment Gauge")
                gauge_chart = create_gauge_chart(prediction_proba[1])
                st.plotly_chart(gauge_chart, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Personalized Recommendations")
                
                recommendations = []
                if inputs['BMI'] > 25:
                    recommendations.append("🏃‍♂️ Consider weight management through diet and exercise")
                if inputs['mean_sbp'] > 140:
                    recommendations.append("🩺 Monitor blood pressure regularly")
                if inputs['smoking_status'] != 'non-smoker':
                    recommendations.append("🚭 Consider smoking cessation programs")
                if inputs['physical_activity'] == 'low':
                    recommendations.append("💪 Increase physical activity gradually")
                if inputs['total_cholesterol'] > 5:
                    recommendations.append("🥗 Consider dietary changes to manage cholesterol")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("🎉 Keep up the excellent health habits!")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    else:
        st.error("🚨 Model not loaded. Please check your model file.")
else:
    st.markdown("""
    <div class="prediction-card info-card">
        <h3>👈 Enter patient information</h3>
        <p>Fill in the patient details in the sidebar and click 'Analyze Patient Risk' to get AI-powered predictions</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p>🫀 CardioPredict AI • Powered by Advanced Machine Learning</p>
    <p>⚠️ This tool is for educational purposes only. Always consult with healthcare professionals for medical decisions.</p>
    <p>Last updated: {datetime.now().strftime("%B %Y")}</p>
</div>
""", unsafe_allow_html=True)