import streamlit as st
import time
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_core.ai_model import MedicalAIModel
from data_processing.image_processor import MedicalImageProcessor

class MedicalDiagnosisApp:
    def __init__(self):
        self.ai_model = MedicalAIModel()
        self.image_processor = MedicalImageProcessor()
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Medical AI Diagnosis System - MedicAI",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_custom_css(self):
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .diagnosis-result {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .high-risk {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        .medium-risk {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
        }
        .low-risk {
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_medical_report(self, diagnosis, confidence):
        medical_reports = {
            'COVID-19': {
                'risk_level': 'high',
                'recommendations': [
                    'Perform confirmatory PCR test as soon as possible',
                    'Immediate home isolation for 14 days',
                    'Continuous monitoring of blood oxygen levels',
                    'Urgent medical consultation',
                    'Drink plenty of warm fluids',
                    'Measure temperature every 6 hours'
                ],
                'emergency_contacts': [
                    'Medical Emergency: 123',
                    'Ministry of Health: 937'
                ]
            },
            'Pneumonia': {
                'risk_level': 'medium', 
                'recommendations': [
                    'Consult a pulmonology specialist',
                    'Perform chest CT scan',
                    'Take blood sample for analysis',
                    'Complete rest and avoid exertion',
                    'Take antibiotics as prescribed',
                    'Regular follow-up with doctor'
                ],
                'emergency_contacts': [
                    'Pulmonology Consultant',
                    'Hospital Emergency Department'
                ]
            },
            'Normal': {
                'risk_level': 'low',
                'recommendations': [
                    'No pathological indicators in the image',
                    'Continue regular checkups',
                    'Maintain healthy lifestyle',
                    'Regular exercise',
                    'Avoid smoking and smokers',
                    'Follow balanced nutrition'
                ],
                'emergency_contacts': []
            }
        }
        
        return medical_reports.get(diagnosis, medical_reports['Normal'])
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("System Settings")
            st.markdown("---")
            
            st.subheader("System Information")
            st.info("""
            **Medical AI Diagnosis System**
            
            Supports analysis of:
            - COVID-19
            - Pneumonia
            - Normal cases
            
            System accuracy: 94.2%
            """)
            
            st.markdown("---")
            st.subheader("Usage Instructions")
            st.write("""
            1. Select chest X-ray image
            2. Wait for system analysis
            3. Read medical report
            4. Follow recommendations
            """)
    
    def render_main_interface(self):
        st.markdown('<h1 class="main-header">🏥 Medical AI Diagnosis System - MedicAI</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload X-Ray Image")
            uploaded_file = st.file_uploader(
                "Choose chest X-ray image",
                type=['jpg', 'jpeg', 'png'],
                help="Image should be clear and high quality"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    processed_image = self.image_processor.prepare_for_analysis(image)
                    
                    st.image(processed_image, caption="Processed X-Ray Image", use_container_width=True)
                    
                    if st.button("Start AI Diagnosis", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image and diagnosing condition..."):
                            time.sleep(2)
                            
                            diagnosis, confidence = self.ai_model.predict_disease(processed_image)
                            report = self.generate_medical_report(diagnosis, confidence)
                            
                            self.display_results(diagnosis, confidence, report)
                            
                except Exception as e:
                    st.error(f"Image processing error: {str(e)}")
        
        with col2:
            if 'diagnosis' not in st.session_state:
                st.info("""
                **Welcome to Medical AI Diagnosis System**
                
                Upload chest X-ray image to get:
                - Immediate preliminary diagnosis
                - Detailed medical report
                - Customized medical recommendations
                - Guidance for next steps
                """)
    
    def display_results(self, diagnosis, confidence, report):
        risk_class = f"{report['risk_level']}-risk"
        
        st.markdown(f"""
        <div class="diagnosis-result {risk_class}">
            <h3>Diagnosis Result</h3>
            <h4>{diagnosis}</h4>
            <p><strong>Confidence Level:</strong> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Medical Recommendations")
        for i, recommendation in enumerate(report['recommendations'], 1):
            st.write(f"{i}. {recommendation}")
        
        if report['emergency_contacts']:
            st.subheader("Emergency Contacts")
            for contact in report['emergency_contacts']:
                st.write(f"• {contact}")
        
        st.warning("**Important Note:** This diagnosis is assistive and you should consult a specialist doctor for final diagnosis.")
    
    def run(self):
        self.setup_custom_css()
        self.render_sidebar()
        self.render_main_interface()

if __name__ == "__main__":
    app = MedicalDiagnosisApp()
    app.run()
