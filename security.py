"""
Security and Data Protection Module
وحدة الأمان وحماية البيانات
"""

import hashlib
import secrets
from datetime import datetime, timedelta

class SecurityManager:
    """مدير الأمان وحماية البيانات"""
    
    def __init__(self):
        self.session_timeout = timedelta(hours=1)
    
    def anonymize_patient_data(self, patient_data):
        """إخفاء هوية بيانات المريض"""
        anonymized = patient_data.copy()
        
        # إخفاء المعلومات الشخصية
        if 'name' in anonymized:
            anonymized['name'] = f"Patient_{hashlib.md5(patient_data['name'].encode()).hexdigest()[:8]}"
        
        if 'patient_id' in anonymized:
            anonymized['patient_id'] = f"ANON_{hashlib.md5(patient_data['patient_id'].encode()).hexdigest()[:8]}"
        
        return anonymized
    
    def validate_session(self, session_start):
        """التحقق من صلاحية الجلسة"""
        return datetime.now() - session_start < self.session_timeout
