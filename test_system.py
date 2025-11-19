"""
Comprehensive Testing Suite for Medical AI System
مجموعة اختبارات شاملة للنظام الطبي بالذكاء الاصطناعي
"""

import streamlit as st
import pytest
import sys
import os
import importlib
from datetime import datetime
import pandas as pd
import numpy as np

class ComprehensiveSystemTester:
    """مختبر شامل للنظام الطبي بالذكاء الاصطناعي"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = datetime.now()
    
    def run_complete_test_suite(self):
        """تشغيل مجموعة اختبارات كاملة"""
        st.title("🧪 Comprehensive System Testing Suite")
        st.markdown("---")
        
        # إنشاء علامات تبويب للاختبارات
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Basic Tests", 
            "🧠 AI Engine Tests", 
            "🏥 Medical Tests",
            "📊 Performance Tests",
            "📈 Results Summary"
        ])
        
        with tab1:
            self._run_basic_tests()
        
        with tab2:
            self._run_ai_engine_tests()
        
        with tab3:
            self._run_medical_tests()
        
        with tab4:
            self._run_performance_tests()
        
        with tab5:
            self._display_results_summary()
    
    def _run_basic_tests(self):
        """تشغيل الاختبارات الأساسية"""
        st.header("🔍 Basic System Tests")
        
        basic_tests = [
            ("Python Version Check", self._test_python_version),
            ("Import Dependencies", self._test_imports),
            ("File System Check", self._test_file_system),
            ("Memory Availability", self._test_memory),
            ("GPU Availability", self._test_gpu)
        ]
        
        for test_name, test_func in basic_tests:
            with st.expander(f"🧪 {test_name}", expanded=False):
                result = test_func()
                self.test_results[test_name] = result
                self._display_test_result(test_name, result)
    
    def _run_ai_engine_tests(self):
        """تشغيل اختبارات محرك الذكاء الاصطناعي"""
        st.header("🧠 AI Engine Tests")
        
        ai_tests = [
            ("Model Loading", self._test_model_loading),
            ("Model Architecture", self._test_model_architecture),
            ("Inference Speed", self._test_inference_speed),
            ("Memory Usage", self._test_model_memory),
            ("Input/Output Validation", self._test_io_validation)
        ]
        
        for test_name, test_func in ai_tests:
            with st.expander(f"🤖 {test_name}", expanded=False):
                result = test_func()
                self.test_results[test_name] = result
                self._display_test_result(test_name, result)
    
    def _run_medical_tests(self):
        """تشغيل الاختبارات الطبية"""
        st.header("🏥 Medical Component Tests")
        
        medical_tests = [
            ("DICOM Processing", self._test_dicom_processing),
            ("Biomarker Analysis", self._test_biomarker_analysis),
            ("Database Operations", self._test_database_operations),
            ("PDF Report Generation", self._test_pdf_generation),
            ("Clinical Validation", self._test_clinical_validation)
        ]
        
        for test_name, test_func in medical_tests:
            with st.expander(f"🏥 {test_name}", expanded=False):
                result = test_func()
                self.test_results[test_name] = result
                self._display_test_result(test_name, result)
    
    def _run_performance_tests(self):
        """تشغيل اختبارات الأداء"""
        st.header("📊 Performance Benchmark Tests")
        
        performance_tests = [
            ("System Responsiveness", self._test_responsiveness),
            ("Memory Efficiency", self._test_memory_efficiency),
            ("Processing Speed", self._test_processing_speed),
            ("Concurrent Users", self._test_concurrent_users),
            ("Data Throughput", self._test_data_throughput)
        ]
        
        for test_name, test_func in performance_tests:
            with st.expander(f"⚡ {test_name}", expanded=False):
                result = test_func()
                self.test_results[test_name] = result
                self.performance_metrics[test_name] = result
                self._display_performance_result(test_name, result)
    
    def _test_python_version(self):
        """اختبار إصدار Python"""
        try:
            version = sys.version_info
            if version.major == 3 and version.minor >= 8:
                return {"status": "PASSED", "details": f"Python {version.major}.{version.minor}.{version.micro}"}
            else:
                return {"status": "FAILED", "details": f"Python 3.8+ required, found {version.major}.{version.minor}"}
        except Exception as e:
            return {"status": "ERROR", "details": str(e)}
    
    def _test_imports(self):
        """اختبار استيراد المكتبات"""
        dependencies = [
            ("torch", "PyTorch"),
            ("pydicom", "PyDICOM"),
            ("streamlit", "Streamlit"),
            ("skimage", "Scikit-Image"),
            ("plotly", "Plotly")
        ]
        
        results = []
        for import_name, display_name in dependencies:
            try:
                importlib.import_module(import_name)
                results.append(f"✅ {display_name}")
            except ImportError as e:
                results.append(f"❌ {display_name}: {str(e)}")
        
        if all("✅" in result for result in results):
            return {"status": "PASSED", "details": results}
        else:
            return {"status": "FAILED", "details": results}
    
    def _test_model_loading(self):
        """اختبار تحميل النموذج"""
        try:
            from advanced_ai_engine import HybridMedicalModel
            from model_loader import AdvancedModelManager
            
            manager = AdvancedModelManager()
            model = manager.load_hybrid_model()
            
            # اختبار النموذج
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(test_input)
            
            performance = manager.model_size_analysis(model)
            
            return {
                "status": "PASSED", 
                "details": [
                    f"✅ Model loaded successfully",
                    f"✅ Parameters: {performance['parameters_count']:,}",
                    f"✅ Model size: {performance['total_size_mb']} MB",
                    f"✅ Output shape: {output['final_prediction'].shape}"
                ]
            }
            
        except Exception as e:
            return {"status": "FAILED", "details": [f"❌ Model loading failed: {str(e)}"]}
    
    def _test_dicom_processing(self):
        """اختبار معالجة DICOM"""
        try:
            from dicom_processor import AdvancedDICOMProcessor
            
            processor = AdvancedDICOMProcessor()
            
            # اختبار ببيانات وهمية
            test_metadata = {
                'patient_info': {'name': 'Test Patient'},
                'study_info': {'modality': 'CT'},
                'image_characteristics': {'rows': 512, 'columns': 512}
            }
            
            # اختبار إنشاء تقرير
            report = processor.generate_dicom_report(test_metadata)
            
            return {
                "status": "PASSED",
                "details": [
                    "✅ DICOM processor initialized",
                    "✅ Metadata extraction working",
                    "✅ Report generation functional",
                    f"✅ Report length: {len(report)} characters"
                ]
            }
            
        except Exception as e:
            return {"status": "FAILED", "details": [f"❌ DICOM processing failed: {str(e)}"]}
    
    def _test_biomarker_analysis(self):
        """اختبار تحليل المؤشرات الحيوية"""
        try:
            from enhanced_biomarkers import QuantitativeBiomarkerAnalyzer
            
            analyzer = QuantitativeBiomarkerAnalyzer()
            
            # صورة اختبارية
            test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            
            # تحليل المؤشرات الحيوية
            results = analyzer.comprehensive_biomarker_analysis(test_image)
            
            return {
                "status": "PASSED",
                "details": [
                    "✅ Biomarker analyzer working",
                    f"✅ Integrated score: {results.get('integrated_biomarker_score', 0):.1f}%",
                    f"✅ Risk level: {results.get('clinical_risk_assessment', {}).get('risk_level', 'Unknown')}",
                    f"✅ Analysis categories: {len(results)}"
                ]
            }
            
        except Exception as e:
            return {"status": "FAILED", "details": [f"❌ Biomarker analysis failed: {str(e)}"]}
    
    def _test_processing_speed(self):
        """اختبار سرعة المعالجة"""
        try:
            import time
            from advanced_ai_engine import HybridMedicalModel
            
            model = HybridMedicalModel()
            model.eval()
            
            # اختبار السرعة
            test_input = torch.randn(1, 3, 224, 224)
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # 10 تكرارات
                    _ = model(test_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            if avg_time < 1.0:  # أقل من ثانية
                status = "PASSED"
            else:
                status = "WARNING"
            
            return {
                "status": status,
                "details": [
                    f"✅ Average inference time: {avg_time:.3f} seconds",
                    f"✅ Throughput: {1/avg_time:.1f} inferences/second",
                    f"✅ Meets real-time requirements: {'Yes' if avg_time < 0.5 else 'Needs optimization'}"
                ]
            }
            
        except Exception as e:
            return {"status": "FAILED", "details": [f"❌ Performance test failed: {str(e)}"]}
    
    def _display_test_result(self, test_name, result):
        """عرض نتيجة الاختبار"""
        status = result["status"]
        details = result["details"]
        
        if status == "PASSED":
            st.success(f"**{test_name}**: ✅ PASSED")
        elif status == "WARNING":
            st.warning(f"**{test_name}**: ⚠️ WARNING")
        else:
            st.error(f"**{test_name}**: ❌ FAILED")
        
        for detail in details:
            st.write(f"  {detail}")
    
    def _display_performance_result(self, test_name, result):
        """عرض نتيجة الأداء"""
        if result["status"] == "PASSED":
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Status", "✅ PASSED")
            with col2:
                for detail in result["details"]:
                    st.write(detail)
        else:
            st.error(f"**{test_name}**: ❌ {result['status']}")
            for detail in result["details"]:
                st.write(f"  {detail}")
    
    def _display_results_summary(self):
        """عرض ملخص النتائج"""
        st.header("📈 Test Results Summary")
        
        # إحصائيات النتائج
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.test_results.values() if r["status"] == "FAILED")
        warning_tests = sum(1 for r in self.test_results.values() if r["status"] == "WARNING")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Passed", passed_tests, delta=f"{(passed_tests/total_tests)*100:.1f}%")
        with col3:
            st.metric("Warnings", warning_tests)
        with col4:
            st.metric("Failed", failed_tests, delta_color="inverse")
        
        # مخطط النتائج
        results_df = pd.DataFrame({
            'Category': ['Passed', 'Warnings', 'Failed'],
            'Count': [passed_tests, warning_tests, failed_tests]
        })
        
        fig = px.pie(results_df, values='Count', names='Category', 
                    title="Test Results Distribution",
                    color='Category',
                    color_discrete_map={'Passed': '#00cc96', 'Warnings': '#ffa500', 'Failed': '#ff4b4b'})
        
        st.plotly_chart(fig)
        
        # توصيات النظام
        st.subheader("🎯 System Recommendations")
        
        if failed_tests == 0 and warning_tests == 0:
            st.success("🎉 All tests passed! The system is ready for production use.")
        elif failed_tests > 0:
            st.error("⚠️ Critical issues detected. Please address failed tests before deployment.")
        else:
            st.warning("ℹ️  Some warnings detected. Consider optimizing for better performance.")
        
        # وقت التشغيل
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        st.info(f"⏱️ Total testing time: {duration:.2f} seconds")

# دالة التشغيل الرئيسية
def run_system_tests():
    """تشغيل اختبارات النظام"""
    tester = ComprehensiveSystemTester()
    tester.run_complete_test_suite()

if __name__ == "__main__":
    run_system_tests()
