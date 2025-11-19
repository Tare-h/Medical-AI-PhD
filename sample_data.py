"""
Sample Medical Data for Testing and Demonstration
بيانات طبية نموذجية للاختبار والعرض التوضيحي
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

class MedicalSampleData:
    """فئة لتوليد بيانات طبية نموذجية"""
    
    def __init__(self):
        self.patient_counter = 1
        self.study_counter =  
    
    def generate_sample_patients(self, count: int = 5) -> List[Dict]:
        """توليد بيانات مرضى نموذجية"""
        patients = []
        
        first_names = ["John", "Maria", "Ahmed", "Wei", "Sophie", "Carlos", "Aisha", "James"]
        last_names = ["Smith", "Garcia", "Khan", "Zhang", "Martin", "Lopez", "Johnson", "Brown"]
        conditions = [
            "Normal Chest X-Ray", "Pneumonia", "Lung Nodule", 
            "Fracture", "Cardiomegaly", "Pleural Effusion"
        ]
        
        for i in range(count):
            patient = {
                'patient_id': f"PAT_{self.patient_counter:03d}",
                'name': f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
                'age': np.random.randint(25, 80),
                'gender': np.random.choice(['Male', 'Female']),
                'birth_date': self._random_date(datetime(1940, 1, 1), datetime(2000, 1, 1)),
                'condition': np.random.choice(conditions),
                'created_at': datetime.now()
            }
            patients.append(patient)
            self.patient_counter += 1
        
        return patients
    
    def generate_sample_studies(self, patient_ids: List[str]) -> List[Dict]:
        """توليد دراسات تصوير نموذجية"""
        studies = []
        
        modalities = ["X-Ray", "CT", "MRI", "Mammography", "Ultrasound"]
        body_parts = ["Chest", "Head", "Abdomen", "Extremity", "Spine"]
        physicians = ["Dr. Smith", "Dr. Johnson", "Dr. Garcia", "Dr. Wilson", "Dr. Lee"]
        
        for patient_id in patient_ids:
            for _ in range(np.random.randint(1, 4)):  # 1-3 دراسات لكل مريض
                study = {
                    'study_id': f"STU_{self.study_counter:04d}",
                    'patient_id': patient_id,
                    'modality': np.random.choice(modalities),
                    'body_part': np.random.choice(body_parts),
                    'study_date': self._random_date(datetime(2023, 1, 1), datetime.now()),
                    'referring_physician': np.random.choice(physicians),
                    'description': f"{np.random.choice(body_parts)} {np.random.choice(modalities)} Study",
                    'image_count': np.random.randint(1, 50)
                }
                studies.append(study)
                self.study_counter += 1
        
        return studies
    
    def generate_sample_analyses(self, study_ids: List[str]) -> List[Dict]:
        """توليد تحليلات ذكاء اصطناعي نموذجية"""
        analyses = []
        
        diagnoses = {
            'Normal': {'confidence_range': (0.85, 0.98), 'risk': 'Low'},
            'Benign': {'confidence_range': (0.75, 0.90), 'risk': 'Low'},
            'Suspicious': {'confidence_range': (0.65, 0.85), 'risk': 'Medium'},
            'Malignant': {'confidence_range': (0.70, 0.95), 'risk': 'High'}
        }
        
        for study_id in study_ids:
            diagnosis_type = np.random.choice(list(diagnoses.keys()))
            diagnosis_info = diagnoses[diagnosis_type]
            
            confidence = np.random.uniform(*diagnosis_info['confidence_range'])
            
            analysis = {
                'analysis_id': f"ANA_{len(analyses) + 1:05d}",
                'study_id': study_id,
                'analysis_date': datetime.now(),
                'primary_diagnosis': f"{diagnosis_type} Finding",
                'confidence_score': confidence,
                'risk_level': diagnosis_info['risk'],
                'processing_time': np.random.uniform(1.5, 4.0),
                'model_version': 'Hybrid-CNN-Transformer-v2.1',
                'biomarkers': self._generate_sample_biomarkers(diagnosis_type),
                'recommendations': self._generate_sample_recommendations(diagnosis_type)
            }
            analyses.append(analysis)
        
        return analyses
    
    def _generate_sample_biomarkers(self, diagnosis: str) -> Dict:
        """توليد مؤشرات حيوية نموذجية بناءً على التشخيص"""
        base_biomarkers = {
            'texture_analysis': {
                'entropy': np.random.uniform(1.5, 4.0),
                'contrast': np.random.uniform(0.1, 0.9),
                'homogeneity': np.random.uniform(0.3, 0.95),
                'energy': np.random.uniform(0.1, 0.8)
            },
            'morphological_analysis': {
                'edge_density': np.random.uniform(0.05, 0.35),
                'region_count': np.random.randint(2, 20),
                'solidity': np.random.uniform(0.6, 0.98),
                'circularity': np.random.uniform(0.3, 0.95)
            },
            'intensity_analysis': {
                'mean_intensity': np.random.uniform(80, 180),
                'std_intensity': np.random.uniform(20, 80),
                'skewness': np.random.uniform(-1.5, 1.5),
                'kurtosis': np.random.uniform(-1.0, 3.0)
            }
        }
        
        # تعديل المؤشرات بناءً على التشخيص
        if diagnosis == 'Malignant':
            base_biomarkers['texture_analysis']['entropy'] += 1.0
            base_biomarkers['morphological_analysis']['edge_density'] += 0.1
        elif diagnosis == 'Suspicious':
            base_biomarkers['texture_analysis']['entropy'] += 0.5
        
        return base_biomarkers
    
    def _generate_sample_recommendations(self, diagnosis: str) -> List[str]:
        """توليد توصيات نموذجية بناءً على التشخيص"""
        recommendations = {
            'Normal': [
                "Routine follow-up as per standard protocol",
                "No immediate intervention required"
            ],
            'Benign': [
                "Short-term follow-up recommended (6-12 months)",
                "Consider additional imaging if symptomatic"
            ],
            'Suspicious': [
                "Further diagnostic workup advised",
                "Multidisciplinary consultation recommended",
                "Consider biopsy if clinically indicated"
            ],
            'Malignant': [
                "Urgent specialist referral required",
                "Immediate diagnostic intervention needed",
                "Multidisciplinary tumor board review"
            ]
        }
        
        return recommendations.get(diagnosis, ["Clinical correlation advised"])
    
    def _random_date(self, start: datetime, end: datetime) -> str:
        """توليد تاريخ عشوائي"""
        random_date = start + timedelta(
            seconds=np.random.randint(0, int((end - start).total_seconds()))
        )
        return random_date.strftime('%Y-%m-%d')
    
    def get_complete_sample_dataset(self, patient_count: int = 5) -> Dict[str, Any]:
        """الحصول على مجموعة بيانات نموذجية كاملة"""
        patients = self.generate_sample_patients(patient_count)
        patient_ids = [p['patient_id'] for p in patients]
        
        studies = self.generate_sample_studies(patient_ids)
        study_ids = [s['study_id'] for s in studies]
        
        analyses = self.generate_sample_analyses(study_ids)
        
        return {
            'patients': patients,
            'studies': studies,
            'analyses': analyses,
            'summary': {
                'total_patients': len(patients),
                'total_studies': len(studies),
                'total_analyses': len(analyses),
                'generation_date': datetime.now().isoformat()
            }
        }

# بيانات نموذجية جاهزة للاستخدام
SAMPLE_PATIENTS = [
    {
        'patient_id': 'PAT_001',
        'name': 'John Smith',
        'age': 45,
        'gender': 'Male',
        'condition': 'Normal Chest X-Ray',
        'birth_date': '1978-05-15'
    },
    {
        'patient_id': 'PAT_002',
        'name': 'Maria Garcia', 
        'age': 62,
        'gender': 'Female',
        'condition': 'Suspicious Lung Nodule',
        'birth_date': '1961-11-22'
    },
    {
        'patient_id': 'PAT_003',
        'name': 'Ahmed Khan',
        'age': 38,
        'gender': 'Male', 
        'condition': 'Pneumonia',
        'birth_date': '1985-08-30'
    }
]

SAMPLE_ANALYSES = {
    'normal_case': {
        'primary_diagnosis': 'Normal Tissue',
        'confidence': 0.94,
        'risk_level': 'Low',
        'biomarkers': {
            'texture_entropy': 2.1,
            'edge_density': 0.12,
            'region_count': 5,
            'integrated_score': 92.5
        },
        'recommendations': [
            "Routine follow-up in 12 months",
            "No further action required"
        ]
    },
    'suspicious_case': {
        'primary_diagnosis': 'Suspicious Lesion Detected',
        'confidence': 0.78,
        'risk_level': 'Medium',
        'biomarkers': {
            'texture_entropy': 3.4,
            'edge_density': 0.28,
            'region_count': 12,
            'integrated_score': 65.2
        },
        'recommendations': [
            "Further diagnostic imaging recommended",
            "Multidisciplinary team review",
            "Consider biopsy if clinically indicated"
        ]
    },
    'malignant_case': {
        'primary_diagnosis': 'High Probability of Malignancy',
        'confidence': 0.91,
        'risk_level': 'High',
        'biomarkers': {
            'texture_entropy': 4.2,
            'edge_density': 0.45,
            'region_count': 18,
            'integrated_score': 34.8
        },
        'recommendations': [
            "Urgent specialist referral required",
            "Immediate diagnostic intervention",
            "Multidisciplinary tumor board review"
        ]
    }
}

# مثال على استخدام الكود
if __name__ == "__main__":
    # إنشاء عينات بيانات طبية
    data_generator = MedicalSampleData()
    sample_dataset = data_generator.get_complete_sample_dataset(3)
    
    print("=== Medical Sample Data Generator ===")
    print(f"Generated {sample_dataset['summary']['total_patients']} patients")
    print(f"Generated {sample_dataset['summary']['total_studies']} studies") 
    print(f"Generated {sample_dataset['summary']['total_analyses']} analyses")
    
    print("\n=== Sample Patients ===")
    for patient in sample_dataset['patients']:
        print(f"ID: {patient['patient_id']}, Name: {patient['name']}, Condition: {patient['condition']}")
