"""
Utility Functions for Medical AI System
الدوال المساعدة للنظام الطبي بالذكاء الاصطناعي
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional

def setup_logging():
    """إعداد نظام التسجيل"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_image_file(file) -> bool:
    """التحقق من صحة ملف الصورة"""
    try:
        if file is None:
            return False
        
        # التحقق من نوع الملف
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom']
        if file.type not in allowed_types:
            st.error(f"❌ Unsupported file type: {file.type}")
            return False
        
        # التحقق من حجم الملف (20MB كحد أقصى)
        max_size = 20 * 1024 * 1024  # 20MB
        if file.size > max_size:
            st.error("❌ File size too large (max 20MB)")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"❌ File validation error: {str(e)}")
        return False

def generate_unique_id(prefix: str = "ID") -> str:
    """إنشاء معرف فريد"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_hash = hashlib.md5(f"{prefix}_{timestamp}".encode()).hexdigest()[:8]
    return f"{prefix}_{unique_hash}"

def format_confidence(confidence: float) -> str:
    """تنسيق مستوى الثقة"""
    if confidence >= 0.9:
        return f"🟢 {confidence*100:.1f}% (High)"
    elif confidence >= 0.7:
        return f"🟡 {confidence*100:.1f}% (Medium)"
    else:
        return f"🔴 {confidence*100:.1f}% (Low)"

def calculate_processing_time(start_time: datetime) -> float:
    """حساب وقت المعالجة"""
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    return processing_time

def create_sample_patient_data() -> Dict[str, Any]:
    """إنشاء بيانات مريض نموذجية للاختبار"""
    return {
        'patient_id': generate_unique_id('PAT'),
        'name': 'Test Patient',
        'age': 45,
        'gender': 'Male',
        'referring_doctor': 'Dr. Smith',
        'study_date': datetime.now().strftime('%Y-%m-%d')
    }

def safe_json_serialize(obj: Any) -> Any:
    """تسجيل JSON آمن للكائنات غير القابلة للتسلسل"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def create_performance_metrics(analysis_data: Dict) -> Dict[str, Any]:
    """إنشاء مقاييس أداء للنظام"""
    metrics = {
        'processing_time': analysis_data.get('processing_time', 0),
        'confidence_score': analysis_data.get('ai_diagnosis', {}).get('confidence', 0),
        'biomarker_score': analysis_data.get('biomarkers', {}).get('integrated_biomarker_score', 0),
        'risk_level': analysis_data.get('ai_diagnosis', {}).get('risk_level', 'Unknown'),
        'modality': analysis_data.get('image_metadata', {}).get('modality', 'Unknown')
    }
    return metrics

def setup_streamlit_config():
    """إعداد تكوين Streamlit"""
    st.set_page_config(
        page_title="Medicai-AI-PND Advanced System",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS مخصص
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
