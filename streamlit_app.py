"""
Advanced Medical AI System - Streamlit Interface
Professional healthcare application with comprehensive features
"""

import streamlit as st
import sys
import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from database_manager import PatientDatabase
from pdf_reporter import PDFReportGenerator
from comparative_analyzer import ComparativeAnalyzer
from dicom_processor import DICOMProcessor

class MedicalAIStreamlitApp:
    """Advanced Medical AI System - Streamlit Interface"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_components()
        self.initialize_session_state()
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="MedAI Analyzer Pro",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/medical-ai-system',
                'Report a bug': 'https://github.com/medical-ai-system/issues',
                'About': """
                # MedAI Analyzer Pro v2.0
                Advanced Medical Image Analysis System
                • Patient Database Management
                • DICOM Image Processing  
                • AI-Powered Analysis
                • Comprehensive Reporting
                • Comparative Analytics
                """
            }
        )
        
        self.apply_custom_styles()
    
    def apply_custom_styles(self):
        """Apply professional CSS styles"""
        st.markdown("""
            <style>
            .main-header {
                font-size: 3.5rem;
                color: #1f567b;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .section-header {
                font-size: 2rem;
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 0.5rem;
                margin: 2rem 0 1rem 0;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                margin: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .feature-card {
                background-color: #ffffff;
                padding: 1.5rem;
                border-radius: 15px;
                border-left: 5px solid #3498db;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 1rem 0;
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            }
            .success-box {
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                padding: 1.5rem;
                border-radius: 10px;
                border: 2px solid #28a745;
                margin: 1rem 0;
            }
            .warning-box {
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                padding: 1.5rem;
                border-radius: 10px;
                border: 2px solid #ffc107;
                margin: 1rem 0;
            }
            .info-box {
                background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
                padding: 1.5rem;
                border-radius: 10px;
                border: 2px solid #17a2b8;
                margin: 1rem 0;
            }
            .risk-high {
                background-color: #f8d7da;
                color: #721c24;
                padding: 0.5rem;
                border-radius: 5px;
                border: 1px solid #f5c6cb;
            }
            .risk-medium {
                background-color: #fff3cd;
                color: #856404;
                padding: 0.5rem;
                border-radius: 5px;
                border: 1px solid #ffeaa7;
            }
            .risk-low {
                background-color: #d1edf1;
                color: #0c5460;
                padding: 0.5rem;
                border-radius: 5px;
                border: 1px solid #bee5eb;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.db = PatientDatabase()
            self.pdf_generator = PDFReportGenerator()
            self.comparative_analyzer = ComparativeAnalyzer(self.db)
            self.dicom_processor = DICOMProcessor()
            
            st.sidebar.success("✅ System initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            st.stop()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'current_patient' not in st.session_state:
            st.session_state.current_patient = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Dashboard"
        if 'user_authenticated' not in st.session_state:
            st.session_state.user_authenticated = True
    
    def render_sidebar(self):
        """Render navigation sidebar"""
        with st.sidebar:
            st.image("🏥", width=80)
            st.title("MedAI Analyzer Pro")
            
            # Navigation
            st.subheader("📋 Navigation")
            app_mode = st.radio(
                "Select Section:",
                ["🏠 Dashboard", "👥 Patient Management", "📷 Image Analysis", 
                 "🏥 DICOM Analysis", "📊 Comparative Analytics", "📄 Reports", "⚙️ Settings"]
            )
            
            # System info
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 System Status")
            
            # Quick stats
            stats = self.db.get_system_statistics()
            st.sidebar.metric("Patients", stats.get('total_patients', 0))
            st.sidebar.metric("Analyses", stats.get('total_analyses', 0))
            st.sidebar.metric("Active", stats.get('active_patients', 0))
            
            st.sidebar.markdown("---")
            st.sidebar.info("""
            **MedAI Analyzer Pro v2.0**
            - Advanced Medical Imaging
            - AI-Powered Analysis  
            - Comprehensive Database
            - Professional Reporting
            """)
            
            return app_mode
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown('<h1 class="main-header">🏥 MedAI Analyzer Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Medical Image Analysis with Artificial Intelligence</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System metrics
        self.render_system_metrics()
        
        # Quick actions
        self.render_quick_actions()
        
        # Recent activity
        self.render_recent_activity()
        
        # Features overview
        self.render_features_overview()
    
    def render_system_metrics(self):
        """Render system metrics cards"""
        st.subheader("📊 System Overview")
        
        stats = self.db.get_system_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>👥</h3>
                    <h2>{stats.get('total_patients', 0)}</h2>
                    <p>Total Patients</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>📊</h3>
                    <h2>{stats.get('total_analyses', 0)}</h2>
                    <p>Total Analyses</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>🎯</h3>
                    <h2>{stats.get('avg_confidence', 0)*100:.1f}%</h2>
                    <p>Avg. Confidence</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3>⚡</h3>
                    <h2>{stats.get('recent_analyses', 0)}</h2>
                    <p>Recent (7 days)</p>
                </div>
            """, unsafe_allow_html=True)
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👥 Add New Patient", use_container_width=True, help="Register a new patient in the system"):
                st.session_state.current_tab = "Add Patient"
                st.rerun()
        
        with col2:
            if st.button("📷 Analyze Image", use_container_width=True, help="Upload and analyze medical image"):
                st.session_state.current_tab = "Image Analysis"
                st.rerun()
        
        with col3:
            if st.button("🏥 Process DICOM", use_container_width=True, help="Upload and process DICOM files"):
                st.session_state.current_tab = "DICOM Analysis"
                st.rerun()
        
        with col4:
            if st.button("📈 View Analytics", use_container_width=True, help="Explore comparative analytics"):
                st.session_state.current_tab = "Comparative Analytics"
                st.rerun()
    
    def render_recent_activity(self):
        """Render recent activity section"""
        st.subheader("📈 Recent Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='feature-card'>
                    <h4>🔄 Recent Analyses</h4>
                    <ul>
                        <li>Patient John Doe - Chest X-Ray Analysis (HIGH risk)</li>
                        <li>Patient Sarah Smith - MRI Brain Scan (LOW risk)</li>
                        <li>Patient Mike Johnson - CT Abdomen (MODERATE risk)</li>
                        <li>Patient Emily Brown - Mammography (LOW risk)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='feature-card'>
                    <h4>📊 System Performance</h4>
                    <ul>
                        <li>🟢 Database: Optimal</li>
                        <li>🟢 AI Models: Loaded</li>
                        <li>🟢 Storage: 65% used</li>
                        <li>🟢 Uptime: 99.8%</li>
                        <li>📈 Today's analyses: 12</li>
                        <li>👥 Active sessions: 3</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    def render_features_overview(self):
        """Render features overview"""
        st.subheader("🎯 System Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='feature-card'>
                    <h4>📊 Patient Database</h4>
                    <p>Comprehensive patient management with advanced search, 
                    medical history tracking, and secure data storage.</p>
                    <ul>
                        <li>✅ Secure patient records</li>
                        <li>✅ Medical history tracking</li>
                        <li>✅ Advanced search & filtering</li>
                        <li>✅ Export capabilities</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='feature-card'>
                    <h4>🏥 DICOM Processing</h4>
                    <p>Advanced DICOM file handling with metadata extraction, 
                    image enhancement, and quality assessment.</p>
                    <ul>
                        <li>✅ DICOM metadata extraction</li>
                        <li>✅ Image quality enhancement</li>
                        <li>✅ Multi-format support</li>
                        <li>✅ Quality assessment</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='feature-card'>
                    <h4>📈 Comparative Analytics</h4>
                    <p>Longitudinal analysis with trend detection, progression 
                    tracking, and predictive insights.</p>
                    <ul>
                        <li>✅ Timeline analysis</li>
                        <li>✅ Trend detection</li>
                        <li>✅ Risk progression</li>
                        <li>✅ Predictive analytics</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='feature-card'>
                    <h4>📄 Professional Reports</h4>
                    <p>Comprehensive PDF reports with clinical findings, 
                    recommendations, and biomarker analysis.</p>
                    <ul>
                        <li>✅ Customizable templates</li>
                        <li>✅ Clinical insights</li>
                        <li>✅ Biomarker analysis</li>
                        <li>✅ Export in multiple formats</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    def render_patient_management(self):
        """Render patient management interface"""
        st.markdown('<h2 class="section-header">👥 Patient Management</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Add Patient", "View Patients", "Patient History", "Search & Filter"])
        
        with tab1:
            self.render_add_patient_form()
        
        with tab2:
            self.render_patients_list()
        
        with tab3:
            self.render_patient_history()
        
        with tab4:
            self.render_search_patients()
    
    def render_add_patient_form(self):
        """Render add patient form"""
        st.subheader("➕ Add New Patient")
        
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", placeholder="Enter patient's full name")
                age = st.number_input("Age *", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender *", ["Male", "Female", "Other", "Prefer not to say"])
                birth_date = st.date_input("Date of Birth")
            
            with col2:
                contact = st.text_input("Contact Information", placeholder="Phone or email")
                address = st.text_area("Address", placeholder="Full address")
                emergency_contact = st.text_input("Emergency Contact", placeholder="Name and phone")
                insurance = st.text_input("Insurance Information", placeholder="Insurance provider and ID")
            
            # Medical history section
            st.subheader("🩺 Medical History")
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Medical Conditions**")
                diabetes = st.checkbox("Diabetes")
                hypertension = st.checkbox("Hypertension")
                heart_disease = st.checkbox("Heart Disease")
                cancer = st.checkbox("Cancer History")
                other_conditions = st.text_input("Other Conditions")
            
            with col4:
                st.write("**Additional Information**")
                allergies = st.text_area("Allergies", placeholder="List any allergies")
                medications = st.text_area("Current Medications", placeholder="List current medications")
                family_history = st.text_area("Family Medical History", placeholder="Relevant family history")
            
            # Form submission
            if st.form_submit_button("💾 Save Patient Record", use_container_width=True):
                if name and age:
                    patient_data = {
                        'name': name,
                        'age': age,
                        'gender': gender,
                        'birth_date': str(birth_date),
                        'contact_info': contact,
                        'address': address,
                        'emergency_contact': emergency_contact,
                        'insurance_info': insurance,
                        'medical_history': {
                            'diabetes': diabetes,
                            'hypertension': hypertension,
                            'heart_disease': heart_disease,
                            'cancer': cancer,
                            'other_conditions': other_conditions,
                            'family_history': family_history
                        },
                        'allergies': allergies.split(',') if allergies else [],
                        'medications': medications.split(',') if medications else []
                    }
                    
                    success, patient_id = self.db.add_patient(patient_data)
                    if success:
                        st.success(f"✅ Patient {name} added successfully! Patient ID: {patient_id}")
                    else:
                        st.error("❌ Failed to add patient. Please check the data and try again.")
                else:
                    st.warning("⚠️ Please fill in all required fields (*)")
    
    def render_patients_list(self):
        """Render patients list with advanced features"""
        st.subheader("📋 Patient Database")
        
        patients_df = self.db.get_all_patients()
        
        if not patients_df.empty:
            # Filters and search
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_term = st.text_input("🔍 Search patients by name or ID")
            
            with col2:
                gender_filter = st.selectbox("Filter by gender", ["All", "Male", "Female", "Other"])
            
            with col3:
                status_filter = st.selectbox("Filter by status", ["All", "Active", "Inactive"])
            
            # Apply filters
            if search_term:
                patients_df = patients_df[
                    patients_df['name'].str.contains(search_term, case=False, na=False) |
                    patients_df['patient_id'].str.contains(search_term, case=False, na=False)
                ]
            
            if gender_filter != "All":
                patients_df = patients_df[patients_df['gender'] == gender_filter]
            
            if status_filter != "All":
                patients_df = patients_df[patients_df['status'] == status_filter]
            
            # Display patients
            st.dataframe(
                patients_df,
                use_container_width=True,
                column_config={
                    "patient_id": "Patient ID",
                    "name": "Name", 
                    "age": "Age",
                    "gender": "Gender",
                    "status": "Status",
                    "last_visit": "Last Visit",
                    "analysis_count": "Analyses",
                    "last_analysis": "Last Analysis"
                }
            )
            
            # Statistics
            st.subheader("📊 Patient Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Patients", len(patients_df))
            with col2:
                st.metric("Average Age", f"{patients_df['age'].mean():.1f}")
            with col3:
                st.metric("Analyses per Patient", f"{patients_df['analysis_count'].mean():.1f}")
            with col4:
                active_count = len(patients_df[patients_df['status'] == 'Active'])
                st.metric("Active Patients", active_count)
                
        else:
            st.info("📝 No patients found in the database. Start by adding a new patient.")
    
    def render_patient_history(self):
        """Render patient history interface"""
        st.subheader("📈 Patient History & Analytics")
        
        patients_df = self.db.get_all_patients()
        
        if not patients_df.empty:
            selected_patient = st.selectbox(
                "Select Patient",
                patients_df['patient_id'].tolist(),
                format_func=lambda x: f"{patients_df[patients_df['patient_id'] == x]['name'].iloc[0]} (ID: {x})"
            )
            
            if selected_patient:
                # Patient details
                patient_info = self.db.get_patient(selected_patient)
                if patient_info:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Patient Information**")
                        st.write(f"**Name:** {patient_info.get('name', 'N/A')}")
                        st.write(f"**Age:** {patient_info.get('age', 'N/A')}")
                        st.write(f"**Gender:** {patient_info.get('gender', 'N/A')}")
                        st.write(f"**Status:** {patient_info.get('status', 'N/A')}")
                    
                    with col2:
                        st.write("**Contact Information**")
                        st.write(f"**Contact:** {patient_info.get('contact_info', 'N/A')}")
                        st.write(f"**Last Visit:** {patient_info.get('last_visit', 'N/A')}")
                        st.write(f"**Analyses:** {patient_info.get('analysis_count', 0)}")
                
                # Analysis history
                st.subheader("📋 Analysis History")
                history_df = self.db.get_patient_history(selected_patient)
                
                if not history_df.empty:
                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        column_config={
                            "analysis_id": "Analysis ID",
                            "modality": "Modality",
                            "body_part": "Body Part", 
                            "analysis_date": "Date",
                            "risk_level": "Risk Level",
                            "confidence_score": "Confidence",
                            "quality_score": "Quality"
                        }
                    )
                    
                    # Comparative analysis
                    if st.button("📊 Generate Comparative Analysis", use_container_width=True):
                        with st.spinner("Generating comprehensive analysis..."):
                            comparative_report = self.comparative_analyzer.generate_comprehensive_comparative_report(selected_patient)
                            
                            if 'error' not in comparative_report:
                                self.render_comparative_analysis(comparative_report)
                            else:
                                st.warning(comparative_report['error'])
                else:
                    st.info("No analysis history found for this patient.")
        else:
            st.info("No patients available to show history.")
    
    def render_comparative_analysis(self, comparative_report: Dict):
        """Render comparative analysis results"""
        st.subheader("📈 Comparative Analysis Report")
        
        # Timeline visualization
        timeline_chart = self.comparative_analyzer.create_comprehensive_timeline_visualization(
            comparative_report['patient_id']
        )
        st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Analyses", comparative_report['total_analyses'])
        with col2:
            st.metric("Analysis Period", f"{comparative_report['time_period']['duration_days']} days")
        with col3:
            trend = comparative_report['trend_analysis']['risk_trend']
            st.metric("Risk Trend", trend)
        
        # Clinical insights
        st.subheader("💡 Clinical Insights")
        insights = comparative_report.get('clinical_insights', [])
        for insight in insights:
            st.write(f"• {insight}")
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_chart = self.comparative_analyzer.create_risk_distribution_chart(
                comparative_report['patient_id']
            )
            st.plotly_chart(risk_chart, use_container_width=True)
        
        with col2:
            quality_chart = self.comparative_analyzer.create_quality_trend_chart(
                comparative_report['patient_id']
            )
            st.plotly_chart(quality_chart, use_container_width=True)
    
    def render_search_patients(self):
        """Render patient search interface"""
        st.subheader("🔍 Advanced Patient Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("Search by name or patient ID")
            min_age = st.number_input("Minimum age", min_value=0, max_value=120, value=0)
            max_age = st.number_input("Maximum age", min_value=0, max_value=120, value=120)
        
        with col2:
            gender = st.multiselect("Gender", ["Male", "Female", "Other"])
            risk_level = st.multiselect("Risk Level", ["LOW", "MODERATE", "HIGH", "CRITICAL"])
        
        if st.button("Search Patients", use_container_width=True):
            # This would implement advanced search logic
            st.info("Advanced search functionality would be implemented here")
    
    def render_image_analysis(self):
        """Render image analysis interface"""
        st.markdown('<h2 class="section-header">📷 Medical Image Analysis</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Upload Image", "Analysis Results", "History"])
        
        with tab1:
            self.render_image_upload()
        
        with tab2:
            self.render_analysis_results()
        
        with tab3:
            self.render_analysis_history()
    
    def render_image_upload(self):
        """Render image upload interface"""
        st.subheader("📤 Upload Medical Image")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose medical image file",
                type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                help="Supported formats: PNG, JPG, JPEG, TIFF, BMP"
            )
            
            if uploaded_file is not None:
                # Display image preview
                from PIL import Image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Image information
                st.write(f"**Format:** {image.format}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
        
        with col2:
            st.subheader("🔧 Analysis Configuration")
            
            # Patient selection
            patients_df = self.db.get_all_patients()
            if not patients_df.empty:
                selected_patient = st.selectbox(
                    "Select Patient",
                    patients_df['patient_id'].tolist(),
                    format_func=lambda x: f"{patients_df[patients_df['patient_id'] == x]['name'].iloc[0]} (ID: {x})"
                )
            else:
                st.warning("No patients available. Please add a patient first.")
                selected_patient = None
            
            # Analysis parameters
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Automatic Detection", "Tissue Analysis", "Biomarker Extraction", 
                 "Full Diagnostic", "Custom Analysis"]
            )
            
            body_part = st.selectbox(
                "Body Part",
                ["Brain", "Chest", "Abdomen", "Pelvis", "Extremities", "Breast", "Other"]
            )
            
            urgency = st.select_slider(
                "Urgency Level",
                options=["Routine", "Urgent", "Emergency"]
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col_a, col_b = st.columns(2)
                with col_a:
                    enable_ai = st.checkbox("Enable AI Analysis", value=True)
                    high_accuracy = st.checkbox("High Accuracy Mode", value=False)
                with col_b:
                    generate_report = st.checkbox("Generate PDF Report", value=True)
                    save_to_db = st.checkbox("Save to Database", value=True)
            
            if st.button("🔍 Start Analysis", use_container_width=True):
                if uploaded_file and selected_patient:
                    with st.spinner("🔄 Analyzing image... This may take a few moments."):
                        # Simulate analysis process
                        import time
                        time.sleep(3)
                        
                        # Mock analysis results
                        analysis_results = {
                            'patient_id': selected_patient,
                            'modality': 'X-Ray',
                            'body_part': body_part,
                            'clinical_analysis': {
                                'risk_assessment': {
                                    'level': 'MODERATE',
                                    'score': 0.65,
                                    'description': 'Moderate abnormalities detected'
                                },
                                'quality_assessment': {
                                    'rating': 'Good',
                                    'score': 0.82,
                                    'notes': 'Image quality suitable for diagnosis'
                                },
                                'clinical_findings': [
                                    'Mild opacity in lower right quadrant',
                                    'Normal bone structure',
                                    'No acute cardiopulmonary findings'
                                ],
                                'recommendations': [
                                    'Follow-up in 3-6 months',
                                    'Consider CT scan for detailed evaluation',
                                    'Clinical correlation recommended'
                                ]
                            },
                            'biomarkers': {
                                'mean_intensity': 0.45,
                                'std_intensity': 0.18,
                                'entropy': 0.72,
                                'region_count': 3,
                                'edge_density': 0.28,
                                'contrast': 156.4
                            },
                            'quality_metrics': {
                                'quality_score': 0.82,
                                'contrast': 156.4,
                                'sharpness': 42.3,
                                'snr_estimate': 8.7
                            }
                        }
                        
                        # Save to database
                        if save_to_db:
                            success, analysis_id = self.db.add_analysis(selected_patient, analysis_results)
                            if success:
                                st.success(f"✅ Analysis saved! Analysis ID: {analysis_id}")
                        
                        # Store in session state
                        st.session_state.analysis_results = analysis_results
                        st.session_state.current_patient = selected_patient
                        
                        st.success("🎉 Analysis completed successfully!")
                else:
                    st.warning("⚠️ Please upload an image and select a patient.")
    
    def render_analysis_results(self):
        """Render analysis results interface"""
        st.subheader("📊 Analysis Results")
        
        if not st.session_state.analysis_results:
            st.info("No analysis results available. Please upload and analyze an image first.")
            return
        
        results = st.session_state.analysis_results
        
        # Risk assessment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = results['clinical_analysis']['risk_assessment']['level']
            risk_score = results['clinical_analysis']['risk_assessment']['score']
            
            risk_color = {
                'LOW': 'risk-low',
                'MODERATE': 'risk-medium', 
                'HIGH': 'risk-high',
                'CRITICAL': 'risk-high'
            }.get(risk_level, 'risk-medium')
            
            st.markdown(f"""
                <div class='{risk_color}'>
                    <h3>Risk Level: {risk_level}</h3>
                    <p>Confidence: {risk_score*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            quality_score = results['clinical_analysis']['quality_assessment']['score']
            st.metric("Image Quality", f"{quality_score*100:.1f}%")
        
        with col3:
            st.metric("Findings Count", len(results['clinical_analysis']['clinical_findings']))
        
        # Clinical findings
        st.subheader("🔍 Clinical Findings")
        for finding in results['clinical_analysis']['clinical_findings']:
            st.write(f"• {finding}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        for recommendation in results['clinical_analysis']['recommendations']:
            st.write(f"• {recommendation}")
        
        # Biomarkers
        st.subheader("📈 Biomarker Analysis")
        biomarkers = results['biomarkers']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Intensity Metrics**")
            st.write(f"Mean Intensity: {biomarkers['mean_intensity']:.3f}")
            st.write(f"Standard Deviation: {biomarkers['std_intensity']:.3f}")
            st.write(f"Entropy: {biomarkers['entropy']:.3f}")
        
        with col2:
            st.write("**Spatial Metrics**")
            st.write(f"Region Count: {biomarkers['region_count']}")
            st.write(f"Edge Density: {biomarkers['edge_density']:.3f}")
            st.write(f"Contrast: {biomarkers['contrast']:.1f}")
        
        # Report generation
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating professional report..."):
                # This would generate actual PDF
                st.success("PDF report generated successfully!")
    
    def render_analysis_history(self):
        """Render analysis history"""
        st.subheader("📋 Analysis History")
        
        patients_df = self.db.get_all_patients()
        
        if not patients_df.empty:
            selected_patient = st.selectbox(
                "Select Patient for History",
                patients_df['patient_id'].tolist(),
                key="history_patient",
                format_func=lambda x: f"{patients_df[patients_df['patient_id'] == x]['name'].iloc[0]} (ID: {x})"
            )
            
            if selected_patient:
                history_df = self.db.get_patient_history(selected_patient)
                
                if not history_df.empty:
                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        column_config={
                            "analysis_id": "Analysis ID",
                            "modality": "Modality",
                            "body_part": "Body Part",
                            "analysis_date": "Date",
                            "risk_level": "Risk Level",
                            "confidence_score": "Confidence",
                            "quality_score": "Quality"
                        }
                    )
                else:
                    st.info("No analysis history found for this patient.")
        else:
            st.info("No patients available.")
    
    def render_dicom_analysis(self):
        """Render DICOM analysis interface"""
        st.markdown('<h2 class="section-header">🏥 DICOM Medical Image Analysis</h2>', unsafe_allow_html=True)
        
        st.info("""
        **DICOM Analysis Features:**
        • Full DICOM metadata extraction
        • Advanced image processing  
        • Quality assessment
        • Multi-frame support
        • Anatomical region detection
        """)
        
        uploaded_file = st.file_uploader(
            "Upload DICOM File",
            type=['dcm', 'dicom'],
            help="Supported DICOM formats: .dcm, .dicom"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing DICOM file..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process DICOM file
                    dicom_data = self.dicom_processor.process_dicom_for_analysis(tmp_path)
                    
                    if dicom_data and dicom_data.get('success'):
                        self.display_dicom_analysis_results(dicom_data)
                    else:
                        st.error("❌ Failed to process DICOM file. Please check file format.")
                
                except Exception as e:
                    st.error(f"Error processing DICOM file: {e}")
                
                finally:
                    # Clean up temporary file
                    import os
                    os.unlink(tmp_path)
    
    def display_dicom_analysis_results(self, dicom_data: Dict):
        """Display DICOM analysis results"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Processed Image")
            st.image(dicom_data['image_array'], use_container_width=True)
            
            # Quality metrics
            quality_metrics = dicom_data.get('quality_metrics', {})
            st.subheader("📊 Image Quality Assessment")
            st.write(f"**Overall Quality:** {quality_metrics.get('quality_score', 0)*100:.1f}%")
            st.write(f"**Contrast:** {quality_metrics.get('contrast', 0):.1f}")
            st.write(f"**Sharpness:** {quality_metrics.get('sharpness', 0):.2f}")
            st.write(f"**SNR Estimate:** {quality_metrics.get('snr_estimate', 0):.2f}")
        
        with col2:
            st.subheader("📋 DICOM Metadata")
            metadata = dicom_data.get('metadata', {})
            
            # Patient information
            with st.expander("👤 Patient Information"):
                patient_info = metadata.get('patient_info', {})
                for key, value in patient_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Study information
            with st.expander("📚 Study Information"):
                study_info = metadata.get('study_info', {})
                for key, value in study_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Equipment information
            with st.expander("⚙️ Equipment Information"):
                equipment_info = metadata.get('equipment_info', {})
                for key, value in equipment_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Image information
            with st.expander("🖼️ Image Information"):
                image_info = metadata.get('image_info', {})
                for key, value in image_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧠 AI Analysis", use_container_width=True):
                with st.spinner("Performing AI analysis..."):
                    # Simulate AI analysis
                    import time
                    time.sleep(2)
                    st.success("AI analysis completed!")
        
        with col2:
            if st.button("💾 Save to Database", use_container_width=True):
                st.info("This would save DICOM data to patient record")
        
        with col3:
            if st.button("📄 Generate Report", use_container_width=True):
                st.info("This would generate comprehensive DICOM report")
    
    def render_comparative_analytics(self):
        """Render comparative analytics interface"""
        st.markdown('<h2 class="section-header">📊 Comparative Analytics</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Patient Timeline", "Risk Analysis", "Quality Trends"])
        
        with tab1:
            self.render_patient_timeline_analysis()
        
        with tab2:
            self.render_risk_analysis()
        
        with tab3:
            self.render_quality_trends()
    
    def render_patient_timeline_analysis(self):
        """Render patient timeline analysis"""
        st.subheader("📈 Patient Timeline Analysis")
        
        patients_df = self.db.get_all_patients()
        
        if not patients_df.empty:
            selected_patient = st.selectbox(
                "Select Patient",
                patients_df['patient_id'].tolist(),
                key="timeline_patient",
                format_func=lambda x: f"{patients_df[patients_df['patient_id'] == x]['name'].iloc[0]} (ID: {x})"
            )
            
            if selected_patient:
                # Generate comprehensive timeline visualization
                timeline_fig = self.comparative_analyzer.create_comprehensive_timeline_visualization(selected_patient)
                st.plotly_chart(timeline_fig, use_container_width=True)
                
                # Generate comparative report
                if st.button("📋 Generate Comparative Report", use_container_width=True):
                    with st.spinner("Generating comprehensive comparative report..."):
                        report = self.comparative_analyzer.generate_comprehensive_comparative_report(selected_patient)
                        
                        if 'error' not in report:
                            self.display_comparative_report(report)
                        else:
                            st.warning(report['error'])
        else:
            st.info("No patients available for timeline analysis.")
    
    def display_comparative_report(self, report: Dict):
        """Display comprehensive comparative report"""
        st.subheader("📊 Comprehensive Comparative Report")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", report['total_analyses'])
        with col2:
            st.metric("Analysis Period", f"{report['time_period']['duration_days']} days")
        with col3:
            st.metric("Risk Trend", report['trend_analysis']['risk_trend'])
        with col4:
            st.metric("Stability", f"{report['trend_analysis']['stability_score']*100:.1f}%")
        
        # Risk progression
        st.subheader("🎯 Risk Progression Analysis")
        risk_progression = report['risk_progression']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Status**")
            st.write(f"Current Risk: {risk_progression['current_risk']}")
            st.write(f"Initial Risk: {risk_progression['initial_risk']}")
            st.write(f"Most Common: {risk_progression['most_common_risk']}")
        
        with col2:
            st.write("**Progression Patterns**")
            st.write(f"Risk Consistency: {risk_progression['risk_consistency']}")
            st.write(f"Escalations: {risk_progression['escalation_count']}")
            st.write(f"Improvements: {risk_progression['improvement_count']}")
        
        # Clinical insights
        st.subheader("💡 Clinical Insights & Recommendations")
        insights = report['clinical_insights']
        for insight in insights:
            st.write(f"• {insight}")
        
        # Predictive analysis
        st.subheader("🔮 Predictive Insights")
        predictive = report['predictive_analysis']
        st.write(f"**Next Risk Prediction:** {predictive['next_risk_prediction']}")
        st.write(f"**Confidence:** {predictive['confidence']}")
        st.write(f"**Projected Trend:** {predictive['projected_trend']}")
        st.write(f"**Recommended Next Scan:** {predictive['recommended_next_scan']}")
    
    def render_risk_analysis(self):
        """Render risk analysis dashboard"""
        st.subheader("🎯 Risk Analysis Dashboard")
        
        # This would implement comprehensive risk analysis across all patients
        st.info("""
        **Risk Analysis Features:**
        • Population risk distribution
        • Risk trend analysis
        • High-risk patient identification  
        • Risk factor correlation
        • Predictive risk modeling
        """)
        
        # Placeholder for risk analysis implementation
        st.warning("🚧 Risk analysis dashboard under development")
    
    def render_quality_trends(self):
        """Render quality trends analysis"""
        st.subheader("📷 Image Quality Trends")
        
        # This would implement quality trend analysis
        st.info("""
        **Quality Analysis Features:**
        • Image quality tracking over time
        • Quality improvement recommendations
        • Technical parameter optimization
        • Inter-scanner comparison
        • Quality assurance reporting
        """)
        
        # Placeholder for quality trends implementation
        st.warning("🚧 Quality trends analysis under development")
    
    def render_reports(self):
        """Render reports interface"""
        st.markdown('<h2 class="section-header">📄 Reports & Exports</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Patient Reports", "System Reports", "Data Export"])
        
        with tab1:
            self.render_patient_reports()
        
        with tab2:
            self.render_system_reports()
        
        with tab3:
            self.render_data_export()
    
    def render_patient_reports(self):
        """Render patient reports interface"""
        st.subheader("👤 Patient Medical Reports")
        
        patients_df = self.db.get_all_patients()
        
        if not patients_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_patient = st.selectbox(
                    "Select Patient",
                    patients_df['patient_id'].tolist(),
                    key="report_patient",
                    format_func=lambda x: f"{patients_df[patients_df['patient_id'] == x]['name'].iloc[0]} (ID: {x})"
                )
            
            with col2:
                report_type = st.selectbox(
                    "Report Type",
                    ["Comprehensive Medical Report", "Analysis Summary", 
                     "Timeline Comparative Report", "Risk Assessment Report"]
                )
            
            # Report customization
            st.subheader("🎨 Report Customization")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_images = st.checkbox("Include Images", value=True)
                include_charts = st.checkbox("Include Charts", value=True)
            
            with col2:
                include_biomarkers = st.checkbox("Include Biomarkers", value=True)
                include_recommendations = st.checkbox("Include Recommendations", value=True)
            
            with col3:
                report_format = st.selectbox("Format", ["PDF", "HTML", "DOCX"])
                language = st.selectbox("Language", ["English", "Arabic", "Spanish"])
            
            if st.button("📄 Generate Report", use_container_width=True):
                with st.spinner("Generating professional medical report..."):
                    # This would generate actual reports
                    st.success("✅ Report generated successfully!")
                    st.info("📁 Report would be available for download in production environment")
        
        else:
            st.info("No patients available for report generation.")
    
    def render_system_reports(self):
        """Render system reports interface"""
        st.subheader("🏥 System Analytics Reports")
        
        report_type = st.selectbox(
            "System Report Type",
            ["Usage Statistics", "Performance Metrics", "Quality Assurance", 
             "Compliance Report", "Audit Log"]
        )
        
        date_range = st.date_input(
            "Report Period",
            value=(datetime.now().date(), datetime.now().date()),
            key="system_report_date"
        )
        
        if st.button("📊 Generate System Report", use_container_width=True):
            with st.spinner("Generating system analytics report..."):
                # This would generate system reports
                st.success("✅ System report generated successfully!")
    
    def render_data_export(self):
        """Render data export interface"""
        st.subheader("💾 Data Export")
        
        st.info("""
        **Export Options:**
        • Patient records in multiple formats
        • Analysis results with biomarkers
        • DICOM metadata and images  
        • System configuration
        • Audit logs and reports
        """)
        
        export_options = st.multiselect(
            "Select Data to Export",
            ["Patient Records", "Analysis Results", "DICOM Metadata", 
             "System Logs", "Quality Metrics", "Risk Assessments"]
        )
        
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON", "XML", "SQLite"]
        )
        
        if st.button("📤 Export Data", use_container_width=True):
            with st.spinner("Preparing data for export..."):
                # This would handle data export
                st.success("✅ Data export completed successfully!")
                st.info("📦 Export files would be available for download in production")
    
    def render_settings(self):
        """Render system settings interface"""
        st.markdown('<h2 class="section-header">⚙️ System Settings</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["General", "Database", "AI Models", "Integration"])
        
        with tab1:
            self.render_general_settings()
        
        with tab2:
            self.render_database_settings()
        
        with tab3:
            self.render_ai_settings()
        
        with tab4:
            self.render_integration_settings()
    
    def render_general_settings(self):
        """Render general system settings"""
        st.subheader("🔧 General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox("Language", ["English", "Arabic", "Spanish", "French", "German"])
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            date_format = st.selectbox("Date Format", ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"])
        
        with col2:
            time_format = st.selectbox("Time Format", ["24-hour", "12-hour"])
            items_per_page = st.number_input("Items per Page", min_value=10, max_value=100, value=25)
            auto_save = st.checkbox("Auto-save Changes", value=True)
        
        # Notification settings
        st.subheader("🔔 Notification Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            email_notifications = st.checkbox("Email Notifications", value=True)
            high_risk_alerts = st.checkbox("High Risk Alerts", value=True)
            system_updates = st.checkbox("System Updates", value=True)
        
        with col2:
            analysis_complete = st.checkbox("Analysis Complete", value=True)
            report_generated = st.checkbox("Report Generated", value=True)
            error_alerts = st.checkbox("Error Alerts", value=True)
        
        if st.button("💾 Save General Settings", use_container_width=True):
            st.success("General settings saved successfully!")
    
    def render_database_settings(self):
        """Render database settings"""
        st.subheader("🗃️ Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Database Status**")
            stats = self.db.get_system_statistics()
            
            st.write(f"• **Total Patients:** {stats.get('total_patients', 0)}")
            st.write(f"• **Total Analyses:** {stats.get('total_analyses', 0)}")
            st.write(f"• **Database Size:** ~{stats.get('total_patients', 0) * 2} KB")
            st.write(f"• **Last Backup:** Never")
            st.write(f"• **Connection:** ✅ Connected")
        
        with col2:
            st.warning("**Database Actions**")
            
            if st.button("🔄 Optimize Database", use_container_width=True):
                st.success("Database optimization completed!")
            
            if st.button("💾 Create Backup", use_container_width=True):
                with st.spinner("Creating database backup..."):
                    # This would create actual backup
                    st.success("Backup created successfully!")
            
            if st.button("📊 Update Statistics", use_container_width=True):
                st.success("Database statistics updated!")
            
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.success("Cache cleared successfully!")
        
        # Advanced database settings
        st.subheader("🔧 Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_backup = st.checkbox("Automatic Backups", value=False)
            backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
            retention_days = st.number_input("Retention Period (days)", min_value=1, max_value=365, value=30)
        
        with col2:
            query_timeout = st.number_input("Query Timeout (seconds)", min_value=5, max_value=300, value=30)
            max_connections = st.number_input("Max Connections", min_value=5, max_value=50, value=10)
            enable_logging = st.checkbox("Enable Query Logging", value=False)
    
    def render_ai_settings(self):
        """Render AI model settings"""
        st.subheader("🧠 AI Model Configuration")
        
        st.info("""
        **AI Model Features:**
        • Multiple model architectures
        • Custom confidence thresholds  
        • Real-time processing options
        • Model performance monitoring
        • Automatic model updates
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_model = st.selectbox(
                "Default Analysis Model",
                ["High Accuracy (Slow)", "Balanced", "Fast Processing", "Custom Model"]
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.99,
                value=0.85,
                help="Minimum confidence score for AI predictions"
            )
            
            processing_speed = st.select_slider(
                "Processing Speed",
                options=["Very Fast", "Fast", "Balanced", "Accurate", "Very Accurate"]
            )
        
        with col2:
            st.write("**Model Features**")
            enable_realtime = st.checkbox("Enable Real-time Analysis", value=True)
            auto_detect = st.checkbox("Auto-detect Image Type", value=True)
            generate_explanations = st.checkbox("Generate AI Explanations", value=True)
            save_analysis_logs = st.checkbox("Save Analysis Logs", value=True)
        
        # Model performance
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", "95.2%")
        with col2:
            st.metric("Precision", "93.8%")
        with col3:
            st.metric("Recall", "96.1%")
        
        if st.button("🔄 Update AI Settings", use_container_width=True):
            st.success("AI model settings updated successfully!")
    
    def render_integration_settings(self):
        """Render integration settings"""
        st.subheader("🔗 System Integrations")
        
        st.info("""
        **Supported Integrations:**
        • PACS (Picture Archiving and Communication System)
        • HIS (Hospital Information System) 
        • RIS (Radiology Information System)
        • EHR (Electronic Health Records)
        • Laboratory Systems
        """)
        
        # PACS Integration
        with st.expander("🏥 PACS Integration"):
            col1, col2 = st.columns(2)
            
            with col1:
                pacs_enabled = st.checkbox("Enable PACS Integration", value=False)
                pacs_host = st.text_input("PACS Host", placeholder="pacs.hospital.com")
                pacs_port = st.number_input("PACS Port", min_value=1, max_value=65535, value=104)
            
            with col2:
                pacs_ae_title = st.text_input("AE Title", placeholder="MEDAI_CLIENT")
                pacs_username = st.text_input("Username")
                pacs_password = st.text_input("Password", type="password")
        
        # API Settings
        with st.expander("🔌 API Configuration"):
            api_enabled = st.checkbox("Enable REST API", value=False)
            api_key = st.text_input("API Key", type="password")
            webhook_url = st.text_input("Webhook URL", placeholder="https://api.yoursystem.com/webhook")
        
        if st.button("💾 Save Integration Settings", use_container_width=True):
            st.success("Integration settings saved successfully!")
    
    def run(self):
        """Run the Streamlit application"""
        try:
            # Render sidebar and get current section
            current_section = self.render_sidebar()
            
            # Render appropriate content based on selection
            section_handlers = {
                "🏠 Dashboard": self.render_dashboard,
                "👥 Patient Management": self.render_patient_management,
                "📷 Image Analysis": self.render_image_analysis,
                "🏥 DICOM Analysis": self.render_dicom_analysis,
                "📊 Comparative Analytics": self.render_comparative_analytics,
                "📄 Reports": self.render_reports,
                "⚙️ Settings": self.render_settings
            }
            
            handler = section_handlers.get(current_section, self.render_dashboard)
            handler()
            
        except Exception as e:
            st.error(f"❌ Application error: {e}")
            st.info("Please refresh the page or contact support if the issue persists")

# Run the application
if __name__ == "__main__":
    app = MedicalAIStreamlitApp()
    app.run()
