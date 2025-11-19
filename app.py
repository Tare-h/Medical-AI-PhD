"""
Medical AI Web Application using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

class MedicalAIApp:
    """
    Interactive web application for medical predictions
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            self.model = joblib.load('saved_models/best_model.pkl')
            self.scaler = joblib.load('saved_models/scaler.pkl')
        except:
            st.warning("Models not found. Please train models first.")
    
    def predict_risk(self, input_data):
        """Predict medical risk"""
        try:
            # Preprocess input
            input_scaled = self.scaler.transform([input_data])
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]
            
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Medical AI Diagnosis System",
            page_icon="🏥",
            layout="wide"
        )
        
        st.title("🏥 Medical AI Diagnosis System")
        st.markdown("---")
        
        # Sidebar for input
        st.sidebar.header("Patient Information")
        
        age = st.sidebar.slider("Age", 18, 100, 50)
        blood_pressure = st.sidebar.slider("Blood Pressure", 80, 200, 120)
        cholesterol = st.sidebar.slider("Cholesterol", 150, 400, 200)
        glucose = st.sidebar.slider("Glucose", 70, 300, 100)
        bmi = st.sidebar.slider("BMI", 18, 40, 25)
        heart_rate = st.sidebar.slider("Heart Rate", 50, 120, 75)
        
        # Prediction
        if st.sidebar.button("Predict Diagnosis"):
            input_data = [age, blood_pressure, cholesterol, glucose, bmi, heart_rate]
            prediction, probabilities = self.predict_risk(input_data)
            
            if prediction is not None:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Diagnosis Result")
                    diagnoses = ['Disease_A', 'Disease_B', 'Healthy']
                    result = diagnoses[prediction]
                    
                    if result == 'Healthy':
                        st.success(f"🎉 Prediction: {result}")
                    else:
                        st.error(f"⚠️ Prediction: {result}")
                
                with col2:
                    st.subheader("Probability Distribution")
                    fig = go.Figure(data=[
                        go.Bar(x=diagnoses, y=probabilities,
                              marker_color=['red', 'orange', 'green'])
                    ])
                    fig.update_layout(title="Diagnosis Probabilities")
                    st.plotly_chart(fig)

if __name__ == "__main__":
    app = MedicalAIApp()
    app.run()
