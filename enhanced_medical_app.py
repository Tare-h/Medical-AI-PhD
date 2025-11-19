"""
نظام التشخيص الطبي بالذكاء الاصطناعي مع دعم الصور الشعاعية
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image as PILImage
import tempfile
import os
import sys

# إضافة المسار للمكتبات المخصصة
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.medical_imaging import MedicalImagingProcessor, SimpleImageProcessor
except ImportError:
    st.error("❌ ملف معالجة الصور غير موجود. تأكد من إنشاء utils/medical_imaging.py")

class EnhancedMedicalApp:
    """التطبيق الطبي المحسن مع تحليل الصور"""
    
    def __init__(self):
        self.setup_app()
        self.load_models()
        
    def setup_app(self):
        """إعداد التطبيق"""
        st.set_page_config(
            page_title="🏥 النظام الطبي الذكي - النسخة المحسنة",
            page_icon="🏥",
            layout="wide"
        )
        
        # التصميم
        st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #2E86AB;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .imaging-section {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # حالة الجلسة
        if 'image_analysis' not in st.session_state:
            st.session_state.image_analysis = None
    
    def load_models(self):
        """تحميل النماذج"""
        try:
            self.clinical_model = joblib.load('saved_models/best_optimized_simple.pkl')
            self.image_processor = MedicalImagingProcessor()
        except:
            self.image_processor = SimpleImageProcessor()
            st.info("ℹ️ استخدام معالج الصور المبسط")
    
    def run(self):
        """تشغيل التطبيق"""
        # العنوان الرئيسي
        st.markdown('<h1 class="main-title">🏥 النظام الطبي الذكي مع تحليل الصور</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # الواجهة الرئيسية
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.show_sidebar()
            
        with col2:
            self.show_main_content()
    
    def show_sidebar(self):
        """الشريط الجانبي"""
        st.markdown("### 📊 البيانات السريرية")
        
        # نموذج إدخال البيانات (مبسط)
        age = st.slider("العمر", 18, 100, 45)
        blood_pressure = st.slider("ضغط الدم", 80, 200, 120)
        cholesterol = st.slider("الكوليسترول", 150, 400, 200)
        
        st.markdown("---")
        st.markdown("### 🖼️ تحليل الصور الشعاعية")
        
        # رفع الصور
        uploaded_file = st.file_uploader(
            "رفع صورة طبية",
            type=['png', 'jpg', 'jpeg'],
            help="يمكن رفع صور الأشعة، الرنين المغناطيسي، أو المسح المقطعي"
        )
        
        if uploaded_file is not None:
            # عرض الصورة
            st.image(uploaded_file, caption="الصورة المرفوعة", use_container_width=True)
            
            # اختيار نوع الصورة
            modality = st.selectbox("نوع الصورة", ["أشعة سينية", "رنين مغناطيسي", "مسح مقطعي"])
            
            # زر التحليل
            if st.button("🔍 تحليل الصورة", use_container_width=True):
                self.analyze_image(uploaded_file, modality)
    
    def analyze_image(self, image_file, modality):
        """تحليل الصورة"""
        with st.spinner("جاري تحليل الصورة الطبية..."):
            try:
                report = self.image_processor.analyze_medical_image(image_file, modality)
                st.session_state.image_analysis = report
                st.success("✅ تم تحليل الصورة بنجاح!")
            except Exception as e:
                st.error(f"❌ فشل تحليل الصورة: {e}")
    
    def show_main_content(self):
        """المحتوى الرئيسي"""
        st.markdown("### 📋 نتائج التحليل")
        
        # عرض نتائج تحليل الصورة
        if st.session_state.image_analysis:
            self.show_image_analysis()
        else:
            self.show_welcome_message()
        
        # التشخيص المتكامل
        st.markdown("---")
        self.show_integrated_diagnosis()
    
    def show_image_analysis(self):
        """عرض تحليل الصورة"""
        report = st.session_state.image_analysis
        
        st.markdown("#### 📊 تقرير تحليل الصورة")
        
        # المؤشرات الرئيسية
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("نوع الصورة", report.get('modality', 'غير معروف'))
        with col2:
            st.metric("مستوى الخطورة", report.get('risk_level', 'غير محدد'))
        with col3:
            biomarkers = report.get('biomarkers', {})
            st.metric("جودة الصورة", biomarkers.get('image_quality', 'غير معروفة'))
        
        # المؤشرات التفصيلية
        if 'biomarkers' in report:
            st.markdown("##### المؤشرات التقنية")
            biomarkers = report['biomarkers']
            for key, value in biomarkers.items():
                if key != 'image_quality':
                    st.write(f"**{key}:** {value:.4f}" if isinstance(value, float) else f"**{key}:** {value}")
        
        # التوصيات
        st.markdown("##### 💡 التوصيات")
        recommendations = report.get('recommendations', [])
        for rec in recommendations:
            st.info(f"• {rec}")
    
    def show_welcome_message(self):
        """رسالة ترحيب"""
        st.markdown("""
        <div style='text-align: center; padding: 40px; color: #666;'>
            <h3>👋 مرحباً بك في النظام الطبي الذكي</h3>
            <p>لبدء التحليل:</p>
            <p>1. أدخل البيانات السريرية في الشريط الجانبي</p>
            <p>2. ارفع صورة طبية للتحليل (اختياري)</p>
            <p>3. انقر على "تحليل الصورة" ثم "تشخيص متكامل"</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_integrated_diagnosis(self):
        """عرض التشخيص المتكامل"""
        st.markdown("### 🎯 التشخيص المتكامل")
        
        if st.button("🚀 تشغيل التشخيص المتكامل", type="primary"):
            if self.clinical_model is None:
                st.error("❌ النموذج السريري غير متوفر. تأكد من تشغيل المرحلة 6 أولاً.")
                return
            
            # بيانات سريرية افتراضية (يمكن استبدالها ببيانات حقيقية)
            clinical_data = [45, 120, 200, 95, 24.5, 72]  # عمر، ضغط، كوليسترول، جلوكوز، BMI، نبض
            
            try:
                # التنبؤ
                prediction = self.clinical_model.predict([clinical_data])[0]
                probabilities = self.clinical_model.predict_proba([clinical_data])[0]
                
                # عرض النتائج
                diagnoses = ['مرض أ', 'مرض ب', 'سليم']
                diagnosis = diagnoses[prediction]
                confidence = probabilities[prediction] * 100
                
                st.markdown(f"#### النتيجة: **{diagnosis}**")
                st.metric("مستوى الثقة", f"{confidence:.1f}%")
                
                # إضافة معلومات الصورة إذا كانت متاحة
                if st.session_state.image_analysis:
                    st.success("📊 تم دمج تحليل الصورة في التشخيص")
                
            except Exception as e:
                st.error(f"❌ خطأ في التشخيص: {e}")
    
    def show_disclaimer(self):
        """إخلاء المسؤولية الطبية"""
        with st.expander("📋 إخلاء المسؤولية الطبية"):
            st.markdown("""
            ### ⚠️ تنويه مهم
            
            **هذا النظام للأغراض التعليمية والبحثية فقط**
            
            **لا يستخدم للتشخيص الطبي الفعلي**
            
            - always consult healthcare professionals
            - not for emergency situations
            - requires clinical validation
            """)

def main():
    """الدالة الرئيسية"""
    try:
        app = EnhancedMedicalApp()
        app.run()
        app.show_disclaimer()
    except Exception as e:
        st.error(f"خطأ في التطبيق: {e}")

if __name__ == "__main__":
    main()
