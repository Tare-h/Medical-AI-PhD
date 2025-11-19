"""
Multi-language Support for Medical AI System
دعم متعدد اللغات للنظام الطبي بالذكاء الاصطناعي
"""

translations = {
    'en': {
        'welcome': 'Medical AI Diagnostic System',
        'upload_image': 'Upload Medical Image',
        'analyze': 'Run AI Analysis',
        'diagnosis': 'AI Diagnosis',
        'confidence': 'Confidence Level',
        'risk': 'Risk Assessment'
    },
    'ar': {
        'welcome': 'نظام التشخيص الطبي بالذكاء الاصطناعي',
        'upload_image': 'رفع صورة طبية',
        'analyze': 'تشغيل التحليل بالذكاء الاصطناعي',
        'diagnosis': 'التشخيص الآلي',
        'confidence': 'مستوى الثقة',
        'risk': 'تقييم المخاطر'
    }
}

class Translator:
    """مترجم للنظام"""
    
    def __init__(self, language='en'):
        self.language = language
    
    def get(self, key):
        """الحصول على الترجمة"""
        return translations.get(self.language, {}).get(key, key)
