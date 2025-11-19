"""
Advanced DICOM Processing for Medical AI System
معالجة متقدمة لملفات DICOM للنظام الطبي بالذكاء الاصطناعي
"""

import pydicom
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import os
import tempfile

class AdvancedDICOMProcessor:
    """معالج متقدم لملفات DICOM مع استخراج البيانات الوصفية"""
    
    def __init__(self):
        self.supported_transfer_syntaxes = {
            '1.2.840.10008.1.2': 'Implicit VR Little Endian',
            '1.2.840.10008.1.2.1': 'Explicit VR Little Endian', 
            '1.2.840.10008.1.2.2': 'Explicit VR Big Endian',
            '1.2.840.10008.1.2.4.50': 'JPEG Baseline',
            '1.2.840.10008.1.2.4.51': 'JPEG Extended',
            '1.2.840.10008.1.2.4.57': 'JPEG Lossless',
            '1.2.840.10008.1.2.4.70': 'JPEG Lossless',
        }
    
    def load_dicom_file(self, file_path: str) -> Optional[pydicom.Dataset]:
        """تحميل ملف DICOM مع معالجة الأخطاء"""
        try:
            # التحقق من أن الملف موجود
            if not os.path.exists(file_path):
                st.error(f"❌ File not found: {file_path}")
                return None
            
            # محاولة قراءة ملف DICOM
            dataset = pydicom.dcmread(file_path, force=True)
            
            # التحقق من أن الملف هو DICOM فعلاً
            if not hasattr(dataset, 'SOPClassUID'):
                st.error("❌ Not a valid DICOM file")
                return None
            
            st.success(f"✅ Successfully loaded DICOM file: {os.path.basename(file_path)}")
            return dataset
            
        except Exception as e:
            st.error(f"❌ Failed to load DICOM file: {str(e)}")
            return None
    
    def extract_comprehensive_metadata(self, dataset: pydicom.Dataset) -> Dict[str, Any]:
        """استخراج بيانات وصفية شاملة من ملف DICOM"""
        metadata = {
            'basic_info': {},
            'patient_info': {},
            'study_info': {}, 
            'series_info': {},
            'image_characteristics': {},
            'equipment_info': {},
            'dicom_tags': {}
        }
        
        try:
            # المعلومات الأساسية
            metadata['basic_info'] = {
                'sop_class_uid': self._get_dicom_value(dataset, 'SOPClassUID'),
                'sop_instance_uid': self._get_dicom_value(dataset, 'SOPInstanceUID'),
                'transfer_syntax': self._get_dicom_value(dataset, 'TransferSyntaxUID'),
                'transfer_syntax_name': self.supported_transfer_syntaxes.get(
                    self._get_dicom_value(dataset, 'TransferSyntaxUID'), 'Unknown'
                ),
                'file_size': os.path.getsize(dataset.filename) if hasattr(dataset, 'filename') else 0
            }
            
            # معلومات المريض
            metadata['patient_info'] = {
                'patient_id': self._get_dicom_value(dataset, 'PatientID'),
                'patient_name': self._get_dicom_value(dataset, 'PatientName'),
                'patient_birth_date': self._get_dicom_value(dataset, 'PatientBirthDate'),
                'patient_sex': self._get_dicom_value(dataset, 'PatientSex'),
                'patient_age': self._get_dicom_value(dataset, 'PatientAge'),
                'patient_weight': self._get_dicom_value(dataset, 'PatientWeight'),
                'patient_size': self._get_dicom_value(dataset, 'PatientSize')
            }
            
            # معلومات الدراسة
            metadata['study_info'] = {
                'study_id': self._get_dicom_value(dataset, 'StudyID'),
                'study_date': self._get_dicom_value(dataset, 'StudyDate'),
                'study_time': self._get_dicom_value(dataset, 'StudyTime'),
                'study_description': self._get_dicom_value(dataset, 'StudyDescription'),
                'referring_physician': self._get_dicom_value(dataset, 'ReferringPhysicianName'),
                'accession_number': self._get_dicom_value(dataset, 'AccessionNumber')
            }
            
            # معلومات السلسلة
            metadata['series_info'] = {
                'series_number': self._get_dicom_value(dataset, 'SeriesNumber'),
                'series_date': self._get_dicom_value(dataset, 'SeriesDate'),
                'series_time': self._get_dicom_value(dataset, 'SeriesTime'),
                'series_description': self._get_dicom_value(dataset, 'SeriesDescription'),
                'modality': self._get_dicom_value(dataset, 'Modality'),
                'body_part_examined': self._get_dicom_value(dataset, 'BodyPartExamined'),
                'patient_position': self._get_dicom_value(dataset, 'PatientPosition')
            }
            
            # خصائص الصورة
            metadata['image_characteristics'] = {
                'rows': self._get_dicom_value(dataset, 'Rows'),
                'columns': self._get_dicom_value(dataset, 'Columns'),
                'bits_allocated': self._get_dicom_value(dataset, 'BitsAllocated'),
                'bits_stored': self._get_dicom_value(dataset, 'BitsStored'),
                'high_bit': self._get_dicom_value(dataset, 'HighBit'),
                'pixel_representation': self._get_dicom_value(dataset, 'PixelRepresentation'),
                'samples_per_pixel': self._get_dicom_value(dataset, 'SamplesPerPixel'),
                'photometric_interpretation': self._get_dicom_value(dataset, 'PhotometricInterpretation'),
                'window_center': self._get_dicom_value(dataset, 'WindowCenter'),
                'window_width': self._get_dicom_value(dataset, 'WindowWidth'),
                'rescale_intercept': self._get_dicom_value(dataset, 'RescaleIntercept'),
                'rescale_slope': self._get_dicom_value(dataset, 'RescaleSlope')
            }
            
            # معلومات المعدات
            metadata['equipment_info'] = {
                'manufacturer': self._get_dicom_value(dataset, 'Manufacturer'),
                'manufacturer_model_name': self._get_dicom_value(dataset, 'ManufacturerModelName'),
                'station_name': self._get_dicom_value(dataset, 'StationName'),
                'software_versions': self._get_dicom_value(dataset, 'SoftwareVersions'),
                'device_serial_number': self._get_dicom_value(dataset, 'DeviceSerialNumber')
            }
            
            # استخراج tags إضافية مهمة
            important_tags = [
                'InstanceNumber', 'ImagePositionPatient', 'ImageOrientationPatient',
                'SliceThickness', 'SliceLocation', 'SpacingBetweenSlices',
                'PixelSpacing', 'ConvolutionKernel', 'KVP', 'XRayTubeCurrent',
                'ExposureTime', 'Exposure', 'ContrastBolusAgent'
            ]
            
            for tag in important_tags:
                value = self._get_dicom_value(dataset, tag)
                if value is not None:
                    metadata['dicom_tags'][tag] = value
            
            return metadata
            
        except Exception as e:
            st.error(f"❌ Metadata extraction failed: {str(e)}")
            return metadata
    
    def _get_dicom_value(self, dataset: pydicom.Dataset, tag: str) -> Any:
        """الحصول على قيمة DICOM مع معالجة الآمنة"""
        try:
            if hasattr(dataset, tag):
                value = getattr(dataset, tag)
                if hasattr(value, 'value'):
                    return value.value
                elif hasattr(value, '__len__') and len(value) == 1:
                    return value[0]
                else:
                    return value
            return None
        except:
            return None
    
    def convert_dicom_to_image(self, dataset: pydicom.Dataset) -> Optional[np.ndarray]:
        """تحويل بيانات DICOM إلى صورة قابلة للعرض"""
        try:
            # الحصول على بيانات البكسل
            pixel_data = dataset.pixel_array
            
            # تطبيق rescale إذا كان متوفراً
            if hasattr(dataset, 'RescaleSlope') and hasattr(dataset, 'RescaleIntercept'):
                slope = float(dataset.RescaleSlope)
                intercept = float(dataset.RescaleIntercept)
                pixel_data = pixel_data * slope + intercept
            
            # تطبيق windowing إذا كان متوفراً
            window_center, window_width = self._get_window_parameters(dataset)
            if window_center is not None and window_width is not None:
                pixel_data = self._apply_window_level(pixel_data, window_center, window_width)
            
            # تحويل إلى 8-bit للعرض
            if pixel_data.dtype != np.uint8:
                pixel_data = self._normalize_to_8bit(pixel_data)
            
            # تحويل إلى RGB إذا كانت الصورة grayscale
            if len(pixel_data.shape) == 2:
                pixel_data = cv2.cvtColor(pixel_data, cv2.COLOR_GRAY2RGB)
            
            return pixel_data
            
        except Exception as e:
            st.error(f"❌ DICOM to image conversion failed: {str(e)}")
            return None
    
    def _get_window_parameters(self, dataset: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
        """الحصول على إعدادات النافذة من بيانات DICOM"""
        try:
            window_center = None
            window_width = None
            
            if hasattr(dataset, 'WindowCenter') and hasattr(dataset, 'WindowWidth'):
                if hasattr(dataset.WindowCenter, '__len__'):
                    window_center = float(dataset.WindowCenter[0])
                else:
                    window_center = float(dataset.WindowCenter)
                
                if hasattr(dataset.WindowWidth, '__len__'):
                    window_width = float(dataset.WindowWidth[0])
                else:
                    window_width = float(dataset.WindowWidth)
            
            return window_center, window_width
            
        except:
            return None, None
    
    def _apply_window_level(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """تطبيق window/level على الصورة"""
        try:
            lower = center - width / 2
            upper = center + width / 2
            
            # تطبيق windowing
            windowed = np.clip(image, lower, upper)
            
            # تحويل إلى [0, 255]
            normalized = ((windowed - lower) / (upper - lower)) * 255.0
            return np.clip(normalized, 0, 255).astype(np.uint8)
            
        except:
            return image
    
    def _normalize_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """تطبيع الصورة إلى 8-bit"""
        try:
            # إزالة القيم المتطرفة
            p_low = np.percentile(image, 2)
            p_high = np.percentile(image, 98)
            
            clipped = np.clip(image, p_low, p_high)
            normalized = ((clipped - p_low) / (p_high - p_low)) * 255.0
            
            return np.clip(normalized, 0, 255).astype(np.uint8)
        except:
            return image.astype(np.uint8)
    
    def process_dicom_for_ai(self, dataset: pydicom.Dataset, target_size: Tuple[int, int] = (512, 512)) -> Dict[str, Any]:
        """معالجة DICOM للتحليل بالذكاء الاصطناعي"""
        try:
            # استخراج البيانات الوصفية
            metadata = self.extract_comprehensive_metadata(dataset)
            
            # تحويل إلى صورة
            image_array = self.convert_dicom_to_image(dataset)
            
            if image_array is None:
                st.error("❌ Failed to convert DICOM to image")
                return {}
            
            # تغيير حجم الصورة
            resized_image = cv2.resize(image_array, target_size)
            
            # تحسين الجودة
            enhanced_image = self._enhance_medical_image(resized_image)
            
            # إعداد البيانات للذكاء الاصطناعي
            processed_data = {
                'image_array': enhanced_image,
                'metadata': metadata,
                'original_shape': image_array.shape,
                'processed_shape': enhanced_image.shape,
                'modality': metadata['series_info'].get('modality', 'Unknown'),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            st.success("✅ DICOM processing completed successfully")
            return processed_data
            
        except Exception as e:
            st.error(f"❌ DICOM processing failed: {str(e)}")
            return {}
    
    def _enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """تحسين جودة الصورة الطبية"""
        try:
            # تحويل إلى grayscale إذا لزم الأمر
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # تطبيق CLAHE لتحسين التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # إزالة الضوضاء
            denoised = cv2.medianBlur(enhanced, 3)
            
            # تحسين الحدة
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
            
        except:
            return image
    
    def generate_dicom_report(self, metadata: Dict[str, Any]) -> str:
        """توليد تقرير مفصل عن ملف DICOM"""
        report = []
        report.append("📋 DICOM File Comprehensive Report")
        report.append("=" * 50)
        
        # معلومات المريض
        report.append("\n👤 PATIENT INFORMATION:")
        for key, value in metadata['patient_info'].items():
            if value:
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # معلومات الدراسة
        report.append("\n📊 STUDY INFORMATION:")
        for key, value in metadata['study_info'].items():
            if value:
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # معلومات السلسلة
        report.append("\n🖼️ SERIES INFORMATION:")
        for key, value in metadata['series_info'].items():
            if value:
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # خصائص الصورة
        report.append("\n🎯 IMAGE CHARACTERISTICS:")
        for key, value in metadata['image_characteristics'].items():
            if value is not None:
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # معلومات المعدات
        report.append("\n⚙️ EQUIPMENT INFORMATION:")
        for key, value in metadata['equipment_info'].items():
            if value:
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(report)
    
    def validate_dicom_file(self, dataset: pydicom.Dataset) -> Dict[str, Any]:
        """التحقق من صحة ملف DICOM وإرجاع النتائج"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'completeness_score': 0
        }
        
        try:
            required_tags = [
                'SOPClassUID', 'SOPInstanceUID', 'StudyDate', 'SeriesDate',
                'Modality', 'Rows', 'Columns', 'BitsAllocated', 'BitsStored'
            ]
            
            present_tags = 0
            for tag in required_tags:
                if hasattr(dataset, tag) and getattr(dataset, tag) is not None:
                    present_tags += 1
                else:
                    validation_results['warnings'].append(f"Missing required tag: {tag}")
                    validation_results['is_valid'] = False
            
            # حساب درجة الاكتمال
            validation_results['completeness_score'] = (present_tags / len(required_tags)) * 100
            
            # التحقق من بيانات البكسل
            if not hasattr(dataset, 'pixel_array'):
                validation_results['errors'].append("No pixel data found")
                validation_results['is_valid'] = False
            
            # التحقق من أبعاد الصورة
            if hasattr(dataset, 'Rows') and hasattr(dataset, 'Columns'):
                rows = dataset.Rows
                cols = dataset.Columns
                if rows < 64 or cols < 64:
                    validation_results['warnings'].append(f"Small image dimensions: {rows}x{cols}")
            
            st.info(f"✅ DICOM validation completed - Score: {validation_results['completeness_score']:.1f}%")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results

# دالة مساعدة للاستخدام
@st.cache_resource
def get_dicom_processor():
    """الحصول على معالج DICOM مع التخزين المؤقت"""
    return AdvancedDICOMProcessor()

if __name__ == "__main__":
    processor = AdvancedDICOMProcessor()
    st.success("✅ Advanced DICOM Processor is ready!")
