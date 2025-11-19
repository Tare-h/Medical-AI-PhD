"""
Configuration Settings for Medical AI System
إعدادات التكوين للنظام الطبي بالذكاء الاصطناعي
"""

import os
from typing import Dict, Any

class Config:
    """فئة التكوين الرئيسية للنظام"""
    
    # إعدادات النظام
    APP_NAME = "Medicai-AI-PND"
    VERSION = "2.1.0"
    DEBUG = False
    
    # إعدادات المسارات
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    
    # إعدادات قاعدة البيانات
    DATABASE_PATH = os.path.join(BASE_DIR, "medical_ai_database.db")
    DATABASE_TIMEOUT = 30
    
    # إعدادات الذكاء الاصطناعي
    AI_MODEL_CONFIG = {
        'cnn_backbone': 'resnet50',
        'transformer_dim': 768,
        'transformer_heads': 8,
        'transformer_layers': 6,
        'num_classes': 5,
        'confidence_threshold': 0.7
    }
    
    # إعدادات معالجة الصور
    IMAGE_PROCESSING = {
        'target_size': (512, 512),
        'normalization_mean': [0.485, 0.456, 0.406],
        'normalization_std': [0.229, 0.224, 0.225],
        'enhancement_clip_limit': 2.0,
        'enhancement_grid_size': (8, 8)
    }
    
    # إعدادات المؤشرات الحيوية
    BIOMARKER_THRESHOLDS = {
        'texture_entropy': {'min': 1.5, 'max': 3.5},
        'texture_contrast': {'min': 0.1, 'max': 0.8},
        'edge_density': {'min': 0.05, 'max': 0.3},
        'region_count': {'min': 3, 'max': 15}
    }
    
    # إعدادات الواجهة
    UI_CONFIG = {
        'page_title': "Medicai-AI-PND Advanced System",
        'page_icon': "🩺",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }
    
    @classmethod
    def create_directories(cls):
        """إنشاء المجلدات المطلوبة"""
        directories = [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.REPORTS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# التهيئة
config = Config()
