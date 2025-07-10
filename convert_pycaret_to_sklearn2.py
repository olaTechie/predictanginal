# convert_pycaret_final.py
import pandas as pd
import joblib
from predictor import PyCaretPredictor

def create_and_test_predictor():
    """Create and test the predictor"""
    
    print("🚀 Creating PyCaret predictor...")
    
    # Create the predictor
    predictor = PyCaretPredictor('All_Variables_Model_LightGBM')
    
    if predictor.model is None:
        print("❌ Failed to create predictor")
        return None
    
    # Test with sample data
    print("\n🧪 Testing predictor...")
    test_data = pd.DataFrame({
        'chest_pain': [1.0], 'age': [55], 'sex': ['Male'], 'ethnic': ['South Asian'], 
        'BMI': [25.5], 'smoking_status': ['ex-smoker'], 'physical_activity': ['moderate'], 
        'mean_sbp': [130], 'mean_dbp': [85], 'mean_heart_rate': [70], 'hba1c': [42], 
        'random_glucose': [6.5], 'total_cholesterol': [5.0], 'hdl': [1.3], 'ldl': [3.2], 
        'triglyceride': [1.0], 'Cholesterol_HDL_Ratio': [3.8], 'fam_chd': [1], 
        'chol_lowering': [0], 'has_t1d': [0], 'has_t2d': [0], 'diabetes_status': ['No Diabetes'],
        'treated_hypertension': [1], 'corticosteroid_use': [0], 'creatinine': [60], 
        'blood_urea_nitrogen': [2.8], 'sodium': [14.5], 'potassium': [14], 'glucose': [6.5], 
        'hemoglobin': [12.5], 'hematocrit': [37], 'mean_corpuscular_volume': [89], 
        'mean_corpuscular_hemoglobin': [30], 'mean_corpuscular_hemoglobin_concentration': [33], 
        'white_blood_cell_count': [6], 'red_blood_cell_count': [4], 'platelet_count': [250], 
        'creatine_phosphokinase': [1200], 'ast': [28], 'uric_acid': [145]
    })
    
    # Test prediction
    prediction, probabilities = predictor.predict(test_data)
    
    if prediction is not None:
        print(f"✅ Test prediction successful!")
        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")
        
        # Save the predictor
        joblib.dump(predictor, 'pycaret_predictor.pkl')
        print("✅ Predictor saved as 'pycaret_predictor.pkl'")
        
        return predictor
    else:
        print("❌ Test prediction failed")
        return None

def test_saved_predictor():
    """Test the saved predictor"""
    print("\n🧪 Testing saved predictor...")
    
    try:
        # Load the saved predictor
        predictor = joblib.load('pycaret_predictor.pkl')
        print("✅ Predictor loaded successfully")
        
        # Test data
        test_data = pd.DataFrame({
            'chest_pain': [0.5], 'age': [45], 'sex': ['Female'], 'ethnic': ['White European'], 
            'BMI': [22.1], 'smoking_status': ['non-smoker'], 'physical_activity': ['high'], 
            'mean_sbp': [110], 'mean_dbp': [70], 'mean_heart_rate': [65], 'hba1c': [35], 
            'random_glucose': [5.2], 'total_cholesterol': [4.2], 'hdl': [1.8], 'ldl': [2.1], 
            'triglyceride': [0.7], 'Cholesterol_HDL_Ratio': [2.3], 'fam_chd': [0], 
            'chol_lowering': [0], 'has_t1d': [0], 'has_t2d': [0], 'diabetes_status': ['No Diabetes'],
            'treated_hypertension': [0], 'corticosteroid_use': [0], 'creatinine': [48], 
            'blood_urea_nitrogen': [2.1], 'sodium': [13.8], 'potassium': [13.2], 'glucose': [5.2], 
            'hemoglobin': [13.5], 'hematocrit': [40.2], 'mean_corpuscular_volume': [92.1], 
            'mean_corpuscular_hemoglobin': [31.2], 'mean_corpuscular_hemoglobin_concentration': [34.1], 
            'white_blood_cell_count': [4.8], 'red_blood_cell_count': [4.2], 'platelet_count': [280], 
            'creatine_phosphokinase': [95], 'ast': [21], 'uric_acid': [118]
        })
        
        # Make prediction
        prediction, probabilities = predictor.predict(test_data)
        
        if prediction is not None:
            print(f"✅ Saved predictor test successful!")
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
            return True
        else:
            print("❌ Saved predictor test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing saved predictor: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Converting PyCaret model to standalone predictor...")
    
    # Create the predictor
    predictor = create_and_test_predictor()
    
    if predictor is not None:
        # Test the saved predictor
        test_result = test_saved_predictor()
        
        if test_result:
            print("\n✅ Conversion completed successfully!")
            print("\nFiles created:")
            print("- predictor.py (predictor class definition)")
            print("- pycaret_predictor.pkl (trained predictor)")
            
            print("\n📋 Next steps:")
            print("1. Copy predictor.py to your Streamlit app directory")
            print("2. Copy pycaret_predictor.pkl to your Streamlit app directory") 
            print("3. Update your Streamlit app to import from predictor")
            print("4. Keep PyCaret in your requirements.txt")
        else:
            print("❌ Conversion failed during testing")
    else:
        print("❌ Conversion failed")