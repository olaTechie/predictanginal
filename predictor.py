# predictor.py
import pandas as pd
import numpy as np

class PyCaretPredictor:
    """A class that wraps PyCaret model for prediction"""
    
    def __init__(self, model_path):
        """Initialize with PyCaret model path"""
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the PyCaret model"""
        try:
            from pycaret.classification import load_model
            self.model = load_model(self.model_path)
            print(f"✅ PyCaret model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load PyCaret model: {e}")
            self.model = None
    
    def predict(self, input_data):
        """Make prediction using PyCaret's preprocessing"""
        if self.model is None:
            print("❌ Model not loaded")
            return None, None
        
        try:
            from pycaret.classification import predict_model
            
            # Make a copy to avoid modifying original data
            data_copy = input_data.copy()
            
            # Use PyCaret's predict_model
            pycaret_result = predict_model(self.model, data=data_copy)
            
            # Extract prediction
            prediction = pycaret_result['prediction_label'].iloc[0]
            
            # Extract probability score
            prob_score = pycaret_result['prediction_score'].iloc[0]
            
            # Convert to standard format [prob_class_0, prob_class_1]
            if prediction == 1:
                probabilities = [1 - prob_score, prob_score]
            else:
                probabilities = [prob_score, 1 - prob_score]
            
            return prediction, probabilities
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None, None
    
    def predict_proba(self, input_data):
        """Get prediction probabilities"""
        prediction, probabilities = self.predict(input_data)
        return probabilities
    
    def get_feature_names(self):
        """Get expected feature names"""
        # Return the expected input features
        return [
            'chest_pain', 'age', 'sex', 'ethnic', 'BMI', 'smoking_status', 
            'physical_activity', 'mean_sbp', 'mean_dbp', 'mean_heart_rate', 
            'hba1c', 'random_glucose', 'total_cholesterol', 'hdl', 'ldl', 
            'triglyceride', 'Cholesterol_HDL_Ratio', 'fam_chd', 'chol_lowering', 
            'has_t1d', 'has_t2d', 'diabetes_status', 'treated_hypertension', 
            'corticosteroid_use', 'creatinine', 'blood_urea_nitrogen', 'sodium', 
            'potassium', 'glucose', 'hemoglobin', 'hematocrit', 
            'mean_corpuscular_volume', 'mean_corpuscular_hemoglobin', 
            'mean_corpuscular_hemoglobin_concentration', 'white_blood_cell_count', 
            'red_blood_cell_count', 'platelet_count', 'creatine_phosphokinase', 
            'ast', 'uric_acid'
        ]