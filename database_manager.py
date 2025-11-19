"""
Medical AI Database Manager
إدارة قاعدة البيانات للنظام الطبي الذكي
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os

class PatientDatabase:
    """فئة متكاملة لإدارة قاعدة البيانات الطبية"""
    
    def __init__(self, db_path: str = "medical_patterns.db"):
        self.db_path = db_path
        self.logger = self._setup_logger()
        self._create_tables()
        self._insert_sample_data()
    
    def _setup_logger(self):
        """إعداد نظام التسجيل"""
        logger = logging.getLogger('MedicalDatabase')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_tables(self):
        """إنشاء جميع الجداول اللازمة"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # جدول المرضى
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    birth_date TEXT,
                    condition TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # جدول الدراسات الطبية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    modality TEXT NOT NULL,
                    body_part TEXT,
                    study_date TEXT,
                    referring_physician TEXT,
                    description TEXT,
                    image_count INTEGER,
                    study_status TEXT DEFAULT 'completed',
                    created_at TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
                )
            ''')
            
            # جدول التحليلات الذكية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    study_id TEXT,
                    analysis_date TEXT,
                    primary_diagnosis TEXT,
                    confidence_score REAL,
                    risk_level TEXT,
                    processing_time REAL,
                    model_version TEXT,
                    biomarkers TEXT,  -- JSON format
                    recommendations TEXT,  -- JSON format
                    cnn_confidence REAL,
                    transformer_confidence REAL,
                    fusion_method TEXT,
                    created_at TEXT,
                    FOREIGN KEY (study_id) REFERENCES studies (study_id) ON DELETE CASCADE
                )
            ''')
            
            # جدول الصور الطبية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS medical_images (
                    image_id TEXT PRIMARY KEY,
                    study_id TEXT,
                    image_path TEXT,
                    original_filename TEXT,
                    file_size INTEGER,
                    image_dimensions TEXT,
                    modality TEXT,
                    body_part TEXT,
                    upload_date TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (study_id) REFERENCES studies (study_id) ON DELETE CASCADE
                )
            ''')
            
            # جدول التقارير الطبية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS medical_reports (
                    report_id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    report_type TEXT,
                    report_content TEXT,
                    generated_by TEXT,
                    generated_date TEXT,
                    signed_by TEXT,
                    signed_date TEXT,
                    report_status TEXT DEFAULT 'draft',
                    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
                )
            ''')
            
            # جدول إحصائيات الاستخدام
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_stats (
                    stat_id TEXT PRIMARY KEY,
                    date TEXT,
                    total_analyses INTEGER DEFAULT 0,
                    successful_analyses INTEGER DEFAULT 0,
                    average_confidence REAL DEFAULT 0,
                    most_common_diagnosis TEXT,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
            self.logger.info("✅ تم إنشاء الجداول بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء الجداول: {e}")
            raise
        finally:
            conn.close()
    
    def _get_connection(self):
        """الحصول على اتصال بقاعدة البيانات"""
        return sqlite3.connect(self.db_path)
    
    def _insert_sample_data(self):
        """إدخال بيانات نموذجية للاختبار"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # التحقق إذا كانت البيانات موجودة مسبقاً
            cursor.execute("SELECT COUNT(*) FROM patients")
            patient_count = cursor.fetchone()[0]
            
            if patient_count == 0:
                # إدخال مرضى نموذجيين
                sample_patients = [
                    ('PAT_001', 'John Smith', 45, 'Male', '1978-05-15', 'Normal Chest X-Ray', datetime.now().isoformat(), datetime.now().isoformat()),
                    ('PAT_002', 'Maria Garcia', 62, 'Female', '1961-11-22', 'Suspicious Lung Nodule', datetime.now().isoformat(), datetime.now().isoformat()),
                    ('PAT_003', 'Ahmed Khan', 38, 'Male', '1985-08-30', 'Pneumonia', datetime.now().isoformat(), datetime.now().isoformat()),
                ]
                
                cursor.executemany('''
                    INSERT INTO patients (patient_id, name, age, gender, birth_date, condition, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', sample_patients)
                
                # إدخال دراسات نموذجية
                sample_studies = [
                    ('STU_001', 'PAT_001', 'X-Ray', 'Chest', '2024-01-15', 'Dr. Smith', 'Routine Chest X-Ray', 1, 'completed', datetime.now().isoformat()),
                    ('STU_002', 'PAT_002', 'CT', 'Chest', '2024-01-16', 'Dr. Johnson', 'CT Chest for Nodule', 45, 'completed', datetime.now().isoformat()),
                    ('STU_003', 'PAT_003', 'X-Ray', 'Chest', '2024-01-17', 'Dr. Garcia', 'Chest X-Ray for Pneumonia', 2, 'completed', datetime.now().isoformat()),
                ]
                
                cursor.executemany('''
                    INSERT INTO studies (study_id, patient_id, modality, body_part, study_date, referring_physician, description, image_count, study_status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', sample_studies)
                
                self.logger.info("✅ تم إدخال البيانات النموذجية بنجاح")
            
            conn.commit()
            
        except Exception as e:
            self.logger.warning(f"⚠️ لم يتم إدخال البيانات النموذجية: {e}")
        finally:
            conn.close()
    
    # ██████  ██████   █████  ██████  ██ ███████ ███████ ██████  
    # ██   ██ ██   ██ ██   ██ ██   ██ ██ ██      ██      ██   ██ 
    # ██████  ██████  ███████ ██   ██ ██ █████   █████   ██████  
    # ██   ██ ██   ██ ██   ██ ██   ██ ██ ██      ██      ██   ██ 
    # ██████  ██   ██ ██   ██ ██████  ██ ███████ ███████ ██   ██ 
    
    def add_patient(self, patient_data: Dict) -> bool:
        """إضافة مريض جديد"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            current_time = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO patients 
                (patient_id, name, age, gender, birth_date, condition, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_data.get('patient_id'),
                patient_data.get('name'),
                patient_data.get('age'),
                patient_data.get('gender'),
                patient_data.get('birth_date'),
                patient_data.get('condition'),
                patient_data.get('created_at', current_time),
                current_time
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ تم إضافة المريض: {patient_data.get('patient_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إضافة المريض: {e}")
            return False
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """الحصول على بيانات مريض محدد"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب بيانات المريض: {e}")
            return None
    
    def get_all_patients(self) -> List[Dict]:
        """الحصول على جميع المرضى"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
            patients = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            result = [dict(zip(columns, patient)) for patient in patients]
            
            conn.close()
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب جميع المرضى: {e}")
            return []
    
    def update_patient(self, patient_id: str, update_data: Dict) -> bool:
        """تحديث بيانات مريض"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            values = list(update_data.values())
            values.append(patient_id)
            values.append(datetime.now().isoformat())  # updated_at
            
            cursor.execute(f'''
                UPDATE patients 
                SET {set_clause}, updated_at = ?
                WHERE patient_id = ?
            ''', values)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ تم تحديث المريض: {patient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحديث المريض: {e}")
            return False
    
    # ███████ ████████ ██    ██ ██████  ██ ██████  ███████ 
    # ██         ██    ██    ██ ██   ██ ██ ██   ██ ██      
    # ███████    ██    ██    ██ ██   ██ ██ ██   ██ ███████ 
    #      ██    ██    ██    ██ ██   ██ ██ ██   ██      ██ 
    # ███████    ██     ██████  ██████  ██ ██████  ███████ 
    
    def add_study(self, study_data: Dict) -> bool:
        """إضافة دراسة طبية جديدة"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO studies 
                (study_id, patient_id, modality, body_part, study_date, 
                 referring_physician, description, image_count, study_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                study_data.get('study_id'),
                study_data.get('patient_id'),
                study_data.get('modality'),
                study_data.get('body_part'),
                study_data.get('study_date'),
                study_data.get('referring_physician'),
                study_data.get('description'),
                study_data.get('image_count', 1),
                study_data.get('study_status', 'completed'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ تم إضافة الدراسة: {study_data.get('study_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إضافة الدراسة: {e}")
            return False
    
    def get_studies_by_patient(self, patient_id: str) -> List[Dict]:
        """الحصول على جميع دراسات مريض محدد"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.*, p.name as patient_name 
                FROM studies s 
                JOIN patients p ON s.patient_id = p.patient_id 
                WHERE s.patient_id = ? 
                ORDER BY s.study_date DESC
            ''', (patient_id,))
            
            studies = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            result = [dict(zip(columns, study)) for study in studies]
            
            conn.close()
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب دراسات المريض: {e}")
            return []
    
    def get_all_studies(self) -> List[Dict]:
        """الحصول على جميع الدراسات"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.*, p.name as patient_name 
                FROM studies s 
                JOIN patients p ON s.patient_id = p.patient_id 
                ORDER BY s.study_date DESC
            ''')
            
            studies = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            result = [dict(zip(columns, study)) for study in studies]
            
            conn.close()
            return result
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب جميع الدراسات: {e}")
            return []
    
    #  █████  ███    ██  █████  ██      ██ ███████ ███████ 
    # ██   ██ ████   ██ ██   ██ ██      ██ ██      ██      
    # ███████ ██ ██  ██ ███████ ██      ██ █████   ███████ 
    # ██   ██ ██  ██ ██ ██   ██ ██      ██ ██           ██ 
    # ██   ██ ██   ████ ██   ██ ███████ ██ ███████ ███████ 
    
    def add_analysis(self, analysis_data: Dict) -> bool:
        """إضافة تحليل ذكي جديد"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # تحويل البيانات إلى JSON
            biomarkers_json = json.dumps(analysis_data.get('biomarkers', {}))
            recommendations_json = json.dumps(analysis_data.get('recommendations', []))
            
            cursor.execute('''
                INSERT OR REPLACE INTO analyses 
                (analysis_id, study_id, analysis_date, primary_diagnosis, 
                 confidence_score, risk_level, processing_time, model_version,
                 biomarkers, recommendations, cnn_confidence, transformer_confidence,
                 fusion_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_data.get('analysis_id'),
                analysis_data.get('study_id'),
                analysis_data.get('analysis_date', datetime.now().isoformat()),
                analysis_data.get('primary_diagnosis'),
                analysis_data.get('confidence_score'),
                analysis_data.get('risk_level'),
                analysis_data.get('processing_time'),
                analysis_data.get('model_version'),
                biomarkers_json,
                recommendations_json,
                analysis_data.get('cnn_confidence'),
                analysis_data.get('transformer_confidence'),
                analysis_data.get('fusion_method', 'cross_attention'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ تم إضافة التحليل: {analysis_data.get('analysis_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إضافة التحليل: {e}")
            return False
    
    def get_analysis_by_study(self, study_id: str) -> Optional[Dict]:
        """الحصول على تحليل دراسة محددة"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM analyses WHERE study_id = ?', (study_id,))
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                analysis = dict(zip(columns, result))
                
                # تحويل JSON back to objects
                analysis['biomarkers'] = json.loads(analysis['biomarkers'])
                analysis['recommendations'] = json.loads(analysis['recommendations'])
                
                conn.close()
                return analysis
            
            conn.close()
            return None
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب التحليل: {e}")
            return None
    
    def get_complete_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """الحصول على التاريخ الطبي الكامل للمريض"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # بيانات المريض
            cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
            patient = cursor.fetchone()
            
            if not patient:
                return {}
            
            columns = [desc[0] for desc in cursor.description]
            patient_data = dict(zip(columns, patient))
            
            # الدراسات والتحليلات
            cursor.execute('''
                SELECT s.*, a.* 
                FROM studies s 
                LEFT JOIN analyses a ON s.study_id = a.study_id 
                WHERE s.patient_id = ? 
                ORDER BY s.study_date DESC
            ''', (patient_id,))
            
            studies_with_analyses = cursor.fetchall()
            study_columns = [desc[0] for desc in cursor.description]
            
            studies = []
            for row in studies_with_analyses:
                study = dict(zip(study_columns, row))
                if study['biomarkers']:
                    study['biomarkers'] = json.loads(study['biomarkers'])
                if study['recommendations']:
                    study['recommendations'] = json.loads(study['recommendations'])
                studies.append(study)
            
            conn.close()
            
            return {
                'patient': patient_data,
                'studies': studies,
                'summary': {
                    'total_studies': len(studies),
                    'total_analyses': len([s for s in studies if s.get('analysis_id')]),
                    'latest_study': studies[0]['study_date'] if studies else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب التاريخ الطبي: {e}")
            return {}
    
    # ███████ ████████  █████  ████████ ███████ ███████ ████████ ██  ██████  █████  
    # ██         ██    ██   ██    ██    ██      ██         ██    ██ ██      ██   ██ 
    # ███████    ██    ███████    ██    █████   ███████    ██    ██ ██      ███████ 
    #      ██    ██    ██   ██    ██    ██           ██    ██    ██ ██      ██   ██ 
    # ███████    ██    ██   ██    ██    ███████ ███████    ██    ██  ██████ ██   ██ 
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            # عدد المرضى
            cursor.execute('SELECT COUNT(*) FROM patients')
            stats['total_patients'] = cursor.fetchone()[0]
            
            # عدد الدراسات
            cursor.execute('SELECT COUNT(*) FROM studies')
            stats['total_studies'] = cursor.fetchone()[0]
            
            # عدد التحليلات
            cursor.execute('SELECT COUNT(*) FROM analyses')
            stats['total_analyses'] = cursor.fetchone()[0]
            
            # متوسط الثقة
            cursor.execute('SELECT AVG(confidence_score) FROM analyses')
            stats['average_confidence'] = cursor.fetchone()[0] or 0
            
            # التشخيصات الأكثر شيوعاً
            cursor.execute('''
                SELECT primary_diagnosis, COUNT(*) as count 
                FROM analyses 
                GROUP BY primary_diagnosis 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            stats['common_diagnoses'] = cursor.fetchall()
            
            # التوزيع حسب الوسائط
            cursor.execute('''
                SELECT modality, COUNT(*) as count 
                FROM studies 
                GROUP BY modality 
                ORDER BY count DESC
            ''')
            stats['modality_distribution'] = cursor.fetchall()
            
            conn.close()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جلب الإحصائيات: {e}")
            return {}
    
    def export_patient_data(self, patient_id: str, export_format: str = 'json') -> Optional[str]:
        """تصدير بيانات المريض"""
        try:
            patient_history = self.get_complete_patient_history(patient_id)
            
            if not patient_history:
                return None
            
            if export_format == 'json':
                return json.dumps(patient_history, indent=2, ensure_ascii=False)
            elif export_format == 'csv':
                # يمكن إضافة تحويل لـ CSV إذا needed
                return json.dumps(patient_history, indent=2, ensure_ascii=False)
            else:
                return json.dumps(patient_history, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في تصدير البيانات: {e}")
            return None
    
    def backup_database(self, backup_path: str) -> bool:
        """إنشاء نسخة احتياطية من قاعدة البيانات"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"✅ تم إنشاء نسخة احتياطية في: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء النسخة الاحتياطية: {e}")
            return False

# إنشاء instance للاستخدام الفوري
patient_db = PatientDatabase()

# دالات مساعدة للاستخدام المباشر
def get_patient_db() -> PatientDatabase:
    """الحصول على instance قاعدة البيانات"""
    return patient_db

def initialize_database():
    """تهيئة قاعدة البيانات (للاستخدام في بداية التطبيق)"""
    return PatientDatabase()

# اختبار الوظائف إذا تم تشغيل الملف مباشرة
if __name__ == "__main__":
    db = PatientDatabase()
    
    # اختبار الإحصائيات
    stats = db.get_system_statistics()
    print("📊 إحصائيات النظام:")
    print(f"المرضى: {stats.get('total_patients', 0)}")
    print(f"الدراسات: {stats.get('total_studies', 0)}")
    print(f"التحليلات: {stats.get('total_analyses', 0)}")
    
    # اختبار جلب المرضى
    patients = db.get_all_patients()
    print(f"\n👥 عدد المرضى في النظام: {len(patients)}")
    
    print("✅ تم اختبار قاعدة البيانات بنجاح!")