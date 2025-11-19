"""
Medical AI Diagnosis System - Professional Version
Advanced web application with comprehensive features
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import sys
import os

# Add path for custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ProfessionalMedicalApp:
    """
    Professional Medical AI Diagnosis System with advanced features
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 'Glucose', 'BMI', 'Heart Rate']
        self.diagnosis_labels = ['Disease A', 'Disease B', 'Healthy']
        self.initialize_session_state()
        self.load_models()
        self.setup_page()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_prediction' not in st.session_state:
            st.session_state.current_prediction = None
        if 'patient_data' not in st.session_state:
            st.session_state.patient_data = None
    
    def setup_page(self):
        """Configure Streamlit page with professional styling"""
        st.set_page_config(
            page_title="🏥 Medical AI Diagnosis System",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Professional CSS styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3.5rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #5D5D5D;
            text-align: center;
            margin-bottom: 2rem;
        }
        .diagnosis-card {
            padding: 25px;
            border-radius: 15px;
            margin: 15px 0;
            border-left: 6px solid;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .diagnosis-healthy {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border-color: #28a745;
        }
        .diagnosis-warning {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border-color: #ffc107;
        }
        .diagnosis-danger {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border-color: #dc3545;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin: 10px 0;
        }
        .feature-input {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 12px;
        }
        .history-item {
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid;
            background: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_models(self):
        """Load trained models with error handling"""
        try:
            # Load the trained model
            self.model = joblib.load('saved_models/best_optimized_simple.pkl')
            
            # Try to load scaler, if not create a dummy one
            try:
                self.scaler = joblib.load('saved_models/scaler.pkl')
                st.sidebar.success("✅ Models loaded successfully!")
            except:
                st.sidebar.warning("⚠️ Using model without feature scaling")
                self.scaler = None
                
        except Exception as e:
            st.error(f"""
            ❌ **Model Loading Error**: {e}
            
            Please ensure you have:
            1. Run Phase 6 to train models (`python phase6_simple.py`)
            2. The file `saved_models/best_optimized_simple.pkl` exists
            """)
    
    def create_advanced_input_form(self):
        """Create professional input form with validation"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Patient Demographics")
        
        # Patient information
        col1, col2 = st.sidebar.columns(2)
        with col1:
            age = st.slider("**Age**", 18, 100, 45, 
                          help="Patient age in years")
            gender = st.selectbox("**Gender**", ["Male", "Female", "Other"])
        
        with col2:
            height = st.slider("**Height (cm)**", 140, 220, 170)
            weight = st.slider("**Weight (kg)**", 40, 150, 70)
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Clinical Measurements")
        
        # Clinical measurements with visual indicators
        blood_pressure = st.sidebar.slider(
            "**Blood Pressure (mmHg)**", 80, 200, 120,
            help="Systolic blood pressure"
        )
        
        cholesterol = st.sidebar.slider(
            "**Cholesterol (mg/dL)**", 150, 400, 200,
            help="Total cholesterol level"
        )
        
        glucose = st.sidebar.slider(
            "**Glucose (mg/dL)**", 70, 300, 95,
            help="Fasting blood glucose level"
        )
        
        heart_rate = st.sidebar.slider(
            "**Heart Rate (bpm)**", 50, 120, 72,
            help="Resting heart rate"
        )
        
        # Additional features
        st.sidebar.markdown("### 🩺 Additional Metrics")
        smoking = st.sidebar.select_slider(
            "**Smoking Status**",
            options=["Non-smoker", "Former smoker", "Current smoker"]
        )
        
        activity = st.sidebar.select_slider(
            "**Physical Activity**",
            options=["Sedentary", "Light", "Moderate", "Active", "Very Active"]
        )
        
        # Store all data
        patient_data = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'heart_rate': heart_rate,
            'smoking': smoking,
            'activity': activity,
            'timestamp': datetime.now().isoformat()
        }
        
        return patient_data
    
    def preprocess_features(self, patient_data):
        """Preprocess patient data for model prediction"""
        # Extract main features for model
        main_features = [
            patient_data['age'],
            patient_data['blood_pressure'],
            patient_data['cholesterol'],
            patient_data['glucose'],
            patient_data['bmi'],
            patient_data['heart_rate']
        ]
        
        return np.array(main_features).reshape(1, -1)
    
    def predict_diagnosis(self, features):
        """Make prediction with comprehensive error handling"""
        if self.model is None:
            st.error("❌ Model not loaded. Please check model files.")
            return None, None
        
        try:
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            return prediction, probabilities
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None, None
    
    def display_comprehensive_results(self, prediction, probabilities, patient_data):
        """Display professional results dashboard"""
        
        # Main results section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            self.display_diagnosis_card(prediction, probabilities)
        
        with col2:
            self.display_confidence_meter(probabilities[prediction])
        
        with col3:
            self.display_risk_assessment(prediction, probabilities)
        
        # Detailed analysis section
        st.markdown("---")
        self.display_detailed_analysis(patient_data, probabilities)
        
        # Feature importance and explanations
        st.markdown("---")
        self.display_model_insights()
        
        # Save to history
        self.save_to_history(prediction, probabilities, patient_data)
    
    def display_diagnosis_card(self, prediction, probabilities):
        """Display diagnosis result with professional styling"""
        diagnosis_text = self.diagnosis_labels[prediction]
        confidence = probabilities[prediction] * 100
        
        st.markdown("### 🎯 Diagnosis Result")
        
        if prediction == 2:  # Healthy
            css_class = "diagnosis-healthy"
            icon = "✅"
            status = "LOW RISK"
        elif prediction == 1:  # Disease B
            css_class = "diagnosis-warning"
            icon = "⚠️"
            status = "MEDIUM RISK"
        else:  # Disease A
            css_class = "diagnosis-danger"
            icon = "🚨"
            status = "HIGH RISK"
        
        st.markdown(f"""
        <div class="diagnosis-card {css_class}">
            <h2 style="margin:0; color: inherit;">{icon} {diagnosis_text}</h2>
            <h1 style="margin:10px 0; color: inherit; font-size: 2.5rem;">{confidence:.1f}%</h1>
            <p style="margin:0; font-weight: bold;">Confidence Level</p>
            <p style="margin:5px 0; font-size: 0.9rem;">Status: {status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_confidence_meter(self, confidence):
        """Display confidence level with visual gauge"""
        st.markdown("### 📊 Confidence Level")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_assessment(self, prediction, probabilities):
        """Display risk assessment metrics"""
        st.markdown("### 🚨 Risk Assessment")
        
        risk_levels = {
            0: ("HIGH", "#dc3545"),
            1: ("MEDIUM", "#ffc107"), 
            2: ("LOW", "#28a745")
        }
        
        risk_text, risk_color = risk_levels[prediction]
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2rem; color: {risk_color}; font-weight: bold;">
                {risk_text}
            </div>
            <div style="font-size: 0.9rem; color: #666;">
                Risk Level
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Probability", f"{max(probabilities)*100:.1f}%")
        with col2:
            uncertainty = 1 - max(probabilities)
            st.metric("Uncertainty", f"{uncertainty*100:.1f}%")
    
    def display_detailed_analysis(self, patient_data, probabilities):
        """Display detailed analysis of patient data"""
        st.markdown("### 📈 Detailed Analysis")
        
        # Probability distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                x=self.diagnosis_labels, 
                y=probabilities,
                labels={'x': 'Diagnosis', 'y': 'Probability'},
                color=probabilities,
                color_continuous_scale=['red', 'orange', 'green']
            )
            fig.update_layout(
                title="Diagnosis Probability Distribution",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Patient metrics
            st.markdown("#### Patient Metrics")
            metrics_data = {
                'Metric': ['BMI', 'Blood Pressure', 'Cholesterol', 'Glucose'],
                'Value': [
                    f"{patient_data['bmi']:.1f}",
                    f"{patient_data['blood_pressure']} mmHg",
                    f"{patient_data['cholesterol']} mg/dL", 
                    f"{patient_data['glucose']} mg/dL"
                ],
                'Status': ['Normal' if patient_data['bmi'] < 25 else 'High',
                          'Normal' if patient_data['blood_pressure'] < 120 else 'High',
                          'Normal' if patient_data['cholesterol'] < 200 else 'High',
                          'Normal' if patient_data['glucose'] < 100 else 'High']
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    def display_model_insights(self):
        """Display model insights and feature importance"""
        st.markdown("### 🔍 Model Insights")
        
        if hasattr(self.model, 'feature_importances_'):
            # Feature importance plot
            importances = self.model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                feature_imp_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Feature importance not available for this model type")
        
        # Medical recommendations
        self.display_medical_recommendations()
    
    def display_medical_recommendations(self):
        """Display medical recommendations based on prediction"""
        st.markdown("### 💡 Medical Recommendations")
        
        recommendations = {
            'Healthy': [
                "✅ Continue maintaining healthy lifestyle",
                "✅ Regular exercise and balanced diet", 
                "✅ Annual health check-ups recommended",
                "✅ Monitor key health indicators"
            ],
            'Disease A': [
                "⚠️ Consult with healthcare professional immediately",
                "⚠️ Comprehensive medical evaluation needed",
                "⚠️ Consider lifestyle modifications",
                "⚠️ Regular monitoring of symptoms"
            ],
            'Disease B': [
                "🔸 Schedule appointment with doctor",
                "🔸 Consider preventive measures", 
                "🔸 Monitor symptoms closely",
                "🔸 Lifestyle adjustments recommended"
            ]
        }
        
        # Show general recommendations
        st.markdown("#### General Health Guidelines:")
        general_tips = [
            "• Maintain healthy BMI (18.5-24.9)",
            "• Keep blood pressure below 120/80 mmHg",
            "• Total cholesterol under 200 mg/dL",
            "• Fasting glucose under 100 mg/dL", 
            "• Regular physical activity",
            "• Balanced nutrition",
            "• Avoid smoking and limit alcohol"
        ]
        
        for tip in general_tips:
            st.markdown(f"<div style='margin: 5px 0;'>{tip}</div>", unsafe_allow_html=True)
    
    def display_prediction_history(self):
        """Display prediction history"""
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("### 📋 Prediction History")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Display recent predictions
            for i, record in enumerate(st.session_state.prediction_history[-5:]):
                diagnosis = self.diagnosis_labels[record['prediction']]
                confidence = record['confidence'] * 100
                timestamp = record['timestamp'][11:16]  # Just time
                
                if record['prediction'] == 2:
                    border_color = "#28a745"
                elif record['prediction'] == 1:
                    border_color = "#ffc107"
                else:
                    border_color = "#dc3545"
                
                st.markdown(f"""
                <div class="history-item" style="border-color: {border_color};">
                    <strong>{diagnosis}</strong> - {confidence:.1f}% confidence
                    <br><small>Time: {timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def save_to_history(self, prediction, probabilities, patient_data):
        """Save prediction to history"""
        history_record = {
            'prediction': prediction,
            'confidence': probabilities[prediction],
            'probabilities': probabilities.tolist(),
            'timestamp': patient_data['timestamp'],
            'patient_data': patient_data
        }
        
        st.session_state.prediction_history.append(history_record)
        st.session_state.current_prediction = history_record
        st.session_state.patient_data = patient_data
    
    def display_medical_disclaimer(self):
        """Display medical disclaimer"""
        with st.expander("📋 Important Medical Disclaimer"):
            st.markdown("""
            ### ⚠️ Medical Disclaimer
            
            **This AI system is for educational and research purposes only.**
            
            **Important Notes:**
            - This is not a substitute for professional medical advice
            - Always consult qualified healthcare professionals for diagnosis
            - Do not make medical decisions based solely on this system
            - Results should be interpreted by medical experts
            - Emergency situations require immediate medical attention
            
            **For Emergencies:**
            - Contact emergency services immediately
            - Consult your healthcare provider
            - Follow professional medical guidance
            
            **Normal Medical Ranges:**
            - Blood Pressure: < 120/80 mmHg
            - Cholesterol: < 200 mg/dL
            - Glucose: 70-100 mg/dL (fasting)
            - BMI: 18.5-24.9
            - Heart Rate: 60-100 bpm
            
            *This system demonstrates AI capabilities in healthcare and should not be used for actual medical diagnosis.*
            """)
    
    def run(self):
        """Run the main application"""
        # Header section
        st.markdown('<h1 class="main-header">🏥 Medical AI Diagnosis System</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced AI-Powered Healthcare Analysis</p>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # Main layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Patient Information")
            patient_data = self.create_advanced_input_form()
            
            # Prediction button
            if st.button("🔍 Analyze Patient Data", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing patient data..."):
                    features = self.preprocess_features(patient_data)
                    prediction, probabilities = self.predict_diagnosis(features)
                    
                    if prediction is not None:
                        st.session_state.show_results = True
                        st.session_state.current_prediction = (prediction, probabilities, patient_data)
        
        with col2:
            st.markdown("### Analysis Results")
            
            if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
                prediction, probabilities, patient_data = st.session_state.current_prediction
                self.display_comprehensive_results(prediction, probabilities, patient_data)
            else:
                self.display_welcome_message()
        
        # Additional features
        self.display_prediction_history()
        self.display_medical_disclaimer()

    def display_welcome_message(self):
        """Display welcome message and instructions"""
        st.markdown("""
        <div style='text-align: center; padding: 50px 20px; color: #666;'>
            <h3>👋 Welcome to Medical AI Diagnosis System</h3>
            <p>Enter patient information in the sidebar and click <strong>'Analyze Patient Data'</strong> to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats or information cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h4>🚀 Fast Analysis</h4>
                <p>Get AI-powered diagnosis in seconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Comprehensive</h4>
                <p>Multiple health factors analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h4>🔍 Detailed Insights</h4>
                <p>Complete analysis with recommendations</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function to run the application"""
    try:
        app = ProfessionalMedicalApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please ensure all required files are available and try again.")

if __name__ == "__main__":
    main()
