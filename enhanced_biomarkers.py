"""
Advanced Biomarker Analysis for Medical Imaging
تحليل المؤشرات الحيوية المتقدم للتصوير الطبي
"""

import numpy as np
import cv2
from skimage import feature, measure, filters, segmentation
from scipy import ndimage, stats
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class QuantitativeBiomarkerAnalyzer:
    """محلل كمي متقدم للمؤشرات الحيوية في التصوير الطبي"""
    
    def __init__(self):
        self.biomarkers_history = []
        self.reference_ranges = self._initialize_reference_ranges()
    
    def _initialize_reference_ranges(self) -> Dict:
        """تهيئة النطاقات المرجعية للمؤشرات الحيوية"""
        return {
            'texture_entropy': {'min': 1.5, 'max': 3.5, 'unit': 'bits'},
            'texture_contrast': {'min': 0.1, 'max': 0.8, 'unit': 'ratio'},
            'texture_homogeneity': {'min': 0.3, 'max': 0.9, 'unit': 'ratio'},
            'morphology_edge_density': {'min': 0.05, 'max': 0.3, 'unit': 'ratio'},
            'morphology_region_count': {'min': 3, 'max': 15, 'unit': 'count'},
            'intensity_std': {'min': 25, 'max': 80, 'unit': 'intensity'},
            'intensity_skewness': {'min': -1, 'max': 1, 'unit': 'dimensionless'}
        }
    
    def comprehensive_biomarker_analysis(self, image: np.ndarray) -> Dict:
        """
        تحليل شامل متقدم للمؤشرات الحيوية
        """
        st.info("🔬 Performing Advanced Quantitative Biomarker Analysis...")
        
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        # تطبيق سلسلة التحليلات المتقدمة
        texture_results = self._advanced_texture_analysis(image_gray)
        morphological_results = self._advanced_morphological_analysis(image_gray)
        intensity_results = self._advanced_intensity_analysis(image_gray)
        statistical_results = self._advanced_statistical_analysis(image_gray)
        spatial_results = self._spatial_distribution_analysis(image_gray)
        
        # التحليل المتكامل
        integrated_score = self._calculate_integrated_biomarker_score(
            texture_results, morphological_results, intensity_results
        )
        
        # تقييم المخاطر بناءً على المؤشرات
        risk_assessment = self._assess_clinical_risk(
            texture_results, morphological_results, intensity_results
        )
        
        comprehensive_results = {
            'texture_analysis': texture_results,
            'morphological_analysis': morphological_results,
            'intensity_analysis': intensity_results,
            'statistical_analysis': statistical_results,
            'spatial_analysis': spatial_results,
            'integrated_biomarker_score': integrated_score,
            'clinical_risk_assessment': risk_assessment,
            'biomarker_interpretation': self._interpret_biomarkers(
                texture_results, morphological_results, intensity_results
            )
        }
        
        # حفظ في السجل التاريخي
        self.biomarkers_history.append(comprehensive_results)
        
        st.success("✅ Advanced Biomarker Analysis Completed!")
        return comprehensive_results
    
    def _advanced_texture_analysis(self, image: np.ndarray) -> Dict:
        """تحليل النسيج المتقدم باستخدام GLCM وميزات هارالك"""
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = feature.greycomatrix(image, distances, angles, 
                                      symmetric=True, normed=True)
            
            texture_features = {
                'contrast': np.mean([feature.greycoprops(glcm, 'contrast')[i, j] 
                                   for i in range(len(distances)) 
                                   for j in range(len(angles))]),
                'homogeneity': np.mean([feature.greycoprops(glcm, 'homogeneity')[i, j] 
                                      for i in range(len(distances)) 
                                      for j in range(len(angles))]),
                'energy': np.mean([feature.greycoprops(glcm, 'energy')[i, j] 
                                 for i in range(len(distances)) 
                                 for j in range(len(angles))]),
                'correlation': np.mean([feature.greycoprops(glcm, 'correlation')[i, j] 
                                      for i in range(len(distances)) 
                                      for j in range(len(angles))]),
                'entropy': self._calculate_advanced_entropy(image),
                'variance': np.var(image),
                'smoothness': 1 - 1/(1 + np.var(image)),
                'uniformity': np.sum([(p/255)**2 for p in np.histogram(image, 256)[0]]),
                'dissimilarity': np.mean([feature.greycoprops(glcm, 'dissimilarity')[i, j] 
                                        for i in range(len(distances)) 
                                        for j in range(len(angles))])
            }
            
            # إضافة ميزات هارالك
            haralick_features = self._calculate_haralick_features(image)
            texture_features.update(haralick_features)
            
        except Exception as e:
            st.warning(f"⚠️ Texture analysis limited: {str(e)}")
            texture_features = self._get_default_texture_features()
        
        return texture_features
    
    def _calculate_advanced_entropy(self, image: np.ndarray) -> float:
        """حساب الإنتروبيا المتقدم باستخدام طرق متعددة"""
        # الإنتروبيا التقليدية
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-8))
        
        # إنتروبيا متعددة المستويات
        multi_level_entropy = self._multilevel_entropy(image)
        
        return float(entropy)
    
    def _multilevel_entropy(self, image: np.ndarray, levels: int = 3) -> float:
        """إنتروبيا متعددة المستويات للتحليل الهيكلي"""
        total_entropy = 0
        current_image = image.copy()
        
        for level in range(levels):
            histogram = cv2.calcHist([current_image], [0], None, [256], [0, 256])
            histogram = histogram / histogram.sum()
            level_entropy = -np.sum(histogram * np.log2(histogram + 1e-8))
            total_entropy += level_entropy / (level + 1)
            
            # تقليل دقة الصورة للمستوى التالي
            if level < levels - 1:
                current_image = cv2.resize(current_image, 
                                         (current_image.shape[1]//2, current_image.shape[0]//2))
        
        return total_entropy / levels
    
    def _calculate_haralick_features(self, image: np.ndarray) -> Dict:
        """حساب ميزات هارالك الإضافية"""
        try:
            # تحسين الصورة لميزات هارالك
            normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            normalized_image = normalized_image.astype(np.uint8)
            
            haralick_features = {
                'haralick_asm': 0.0,  # Angular Second Moment
                'haralick_idm': 0.0,  # Inverse Difference Moment
            }
            
            # حساب مبسط لميزات هارالك
            glcm = feature.greycomatrix(normalized_image, [1], [0], symmetric=True, normed=True)
            haralick_features['haralick_asm'] = feature.greycoprops(glcm, 'energy')[0, 0]
            haralick_features['haralick_idm'] = feature.greycoprops(glcm, 'homogeneity')[0, 0]
            
            return haralick_features
        except:
            return {'haralick_asm': 0.0, 'haralick_idm': 0.0}
    
    def _advanced_morphological_analysis(self, image: np.ndarray) -> Dict:
        """تحليل شكلي متقدم باستخدام خوارزميات التجزئة"""
        try:
            # تطبيق Otsu thresholding
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # التنقية المورفولوجية
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)
            
            # توصيف المناطق
            labeled_image = measure.label(cleaned_image)
            regions = measure.regionprops(labeled_image, intensity_image=image)
            
            if len(regions) > 0:
                main_region = max(regions, key=lambda x: x.area)
                
                morphological_features = {
                    'region_count': len(regions),
                    'total_area': sum([r.area for r in regions]),
                    'largest_area': main_region.area,
                    'perimeter': main_region.perimeter,
                    'eccentricity': main_region.eccentricity,
                    'solidity': main_region.solidity,
                    'extent': main_region.extent,
                    'equivalent_diameter': main_region.equivalent_diameter,
                    'edge_density': self._calculate_advanced_edge_density(image),
                    'circularity': (4 * np.pi * main_region.area) / (main_region.perimeter ** 2) 
                                  if main_region.perimeter > 0 else 0,
                    'compactness': (main_region.perimeter ** 2) / (4 * np.pi * main_region.area) 
                                  if main_region.area > 0 else 0
                }
                
                # إضافة ميزات الشكل
                morphological_features.update(self._calculate_shape_features(main_region))
                
            else:
                morphological_features = self._get_default_morphological_features()
                
        except Exception as e:
            st.warning(f"⚠️ Morphological analysis limited: {str(e)}")
            morphological_features = self._get_default_morphological_features()
        
        return morphological_features
    
    def _calculate_advanced_edge_density(self, image: np.ndarray) -> float:
        """حساب كثافة الحواف المتقدم باستخدام كاشفات متعددة"""
        # كاشف Canny
        edges_canny = cv2.Canny(image, 50, 150)
        
        # كاشف Sobel
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edges_sobel = (sobel_magnitude > 50).astype(np.uint8) * 255
        
        # الجمع بين الكاشفات
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        edge_density = np.sum(combined_edges > 0) / (image.shape[0] * image.shape[1])
        
        return float(edge_density)
    
    def _calculate_shape_features(self, region) -> Dict:
        """حساب ميزات الشكل المتقدمة"""
        try:
            # الملامح
            convex_hull = region.convex_image
            convex_area = np.sum(convex_hull)
            
            shape_features = {
                'convexity': region.area / convex_area if convex_area > 0 else 0,
                'elongation': region.major_axis_length / region.minor_axis_length 
                            if region.minor_axis_length > 0 else 0,
                'roundness': (4 * region.area) / (np.pi * (region.major_axis_length ** 2)) 
                           if region.major_axis_length > 0 else 0
            }
            
            return shape_features
        except:
            return {'convexity': 0, 'elongation': 0, 'roundness': 0}
    
    def _advanced_intensity_analysis(self, image: np.ndarray) -> Dict:
        """تحليل شدة متقدم مع إحصائيات متقدمة"""
        flattened = image.flatten()
        
        intensity_features = {
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'min_intensity': np.min(image),
            'max_intensity': np.max(image),
            'median_intensity': np.median(image),
            'intensity_range': np.ptp(image),
            'skewness': float(stats.skew(flattened)),
            'kurtosis': float(stats.kurtosis(flattened)),
            'energy': np.sum(image.astype(np.float64)**2),
            'rms_intensity': np.sqrt(np.mean(image**2)),
            'percentile_25': np.percentile(image, 25),
            'percentile_75': np.percentile(image, 75),
            'iqr_intensity': np.percentile(image, 75) - np.percentile(image, 25)
        }
        
        # إضافة تحليل القمم
        intensity_features.update(self._peak_analysis(image))
        
        return intensity_features
    
    def _peak_analysis(self, image: np.ndarray) -> Dict:
        """تحليل توزيع القمم في الشدة"""
        try:
            histogram, bins = np.histogram(image, bins=256, range=(0, 255))
            peaks, properties = find_peaks(histogram, height=10, distance=10)
            
            peak_features = {
                'peak_count': len(peaks),
                'dominant_peak_height': np.max(properties['peak_heights']) if len(peaks) > 0 else 0,
                'peak_variance': np.var(properties['peak_heights']) if len(peaks) > 1 else 0
            }
            
            return peak_features
        except:
            return {'peak_count': 0, 'dominant_peak_height': 0, 'peak_variance': 0}
    
    def _advanced_statistical_analysis(self, image: np.ndarray) -> Dict:
        """تحليل إحصائي متقدم"""
        flattened = image.flatten()
        
        statistical_features = {
            'variance': np.var(flattened),
            'coefficient_variation': np.std(flattened) / np.mean(flattened) 
                                   if np.mean(flattened) > 0 else 0,
            'entropy': stats.entropy(np.histogram(flattened, bins=256)[0] + 1e-8),
            'uniformity': np.sum((np.histogram(flattened, bins=256)[0] / len(flattened))**2),
            'smoothness': 1 - 1/(1 + np.var(flattened)),
            'third_moment': stats.moment(flattened, moment=3),
            'fourth_moment': stats.moment(flattened, moment=4)
        }
        
        return statistical_features
    
    def _spatial_distribution_analysis(self, image: np.ndarray) -> Dict:
        """تحليل التوزيع المكاني للميزات"""
        try:
            # تقسيم الصورة إلى أرباع
            h, w = image.shape
            quarters = [
                image[0:h//2, 0:w//2],      # الربع العلوي الأيسر
                image[0:h//2, w//2:w],      # الربع العلوي الأيمن
                image[h//2:h, 0:w//2],      # الربع السفلي الأيسر
                image[h//2:h, w//2:w]       # الربع السفلي الأيمن
            ]
            
            quarter_means = [np.mean(q) for q in quarters]
            quarter_stds = [np.std(q) for q in quarters]
            
            spatial_features = {
                'spatial_uniformity': np.std(quarter_means) / (np.mean(quarter_means) + 1e-8),
                'max_quarter_contrast': (max(quarter_means) - min(quarter_means)) / (max(quarter_means) + 1e-8),
                'horizontal_symmetry': abs(quarter_means[0] - quarter_means[1]) / (np.mean(quarter_means) + 1e-8),
                'vertical_symmetry': abs(quarter_means[0] - quarter_means[2]) / (np.mean(quarter_means) + 1e-8)
            }
            
            return spatial_features
        except:
            return self._get_default_spatial_features()
    
    def _calculate_integrated_biomarker_score(self, texture: Dict, morphology: Dict, intensity: Dict) -> float:
        """حساب درجة مؤشر حيوي متكاملة"""
        # أوزان سريرية
        weights = {
            'texture': 0.35,
            'morphology': 0.40,
            'intensity': 0.25
        }
        
        # تسجيل المكونات
        texture_score = self._normalize_score(
            texture.get('entropy', 0), 
            self.reference_ranges['texture_entropy']['min'],
            self.reference_ranges['texture_entropy']['max']
        )
        
        morphology_score = self._normalize_score(
            morphology.get('edge_density', 0),
            self.reference_ranges['morphology_edge_density']['min'],
            self.reference_ranges['morphology_edge_density']['max']
        )
        
        intensity_score = self._normalize_score(
            intensity.get('std_intensity', 0),
            self.reference_ranges['intensity_std']['min'],
            self.reference_ranges['intensity_std']['max']
        )
        
        # حساب الدرجة المتكاملة
        integrated_score = (
            weights['texture'] * texture_score +
            weights['morphology'] * morphology_score +
            weights['intensity'] * intensity_score
        )
        
        return min(integrated_score * 100, 100)
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """تطبيع القيمة إلى نطاق [0,1]"""
        if value < min_val:
            return 0.0
        elif value > max_val:
            return 1.0
        else:
            return (value - min_val) / (max_val - min_val)
    
    def _assess_clinical_risk(self, texture: Dict, morphology: Dict, intensity: Dict) -> Dict:
        """تقييم المخاطر السريرية بناءً على المؤشرات الحيوية"""
        risk_factors = []
        
        # تقييم نسيج
        if texture.get('entropy', 0) > 3.0:
            risk_factors.append("High tissue heterogeneity")
        if texture.get('contrast', 0) > 0.7:
            risk_factors.append("Elevated tissue contrast")
        
        # تقييم شكلي
        if morphology.get('edge_density', 0) > 0.25:
            risk_factors.append("Increased structural complexity")
        if morphology.get('region_count', 0) > 12:
            risk_factors.append("Multiple tissue regions detected")
        
        # تقييم الشدة
        if intensity.get('std_intensity', 0) > 70:
            risk_factors.append("High intensity variability")
        if abs(intensity.get('skewness', 0)) > 0.8:
            risk_factors.append("Abnormal intensity distribution")
        
        # تحديد مستوى الخطورة
        risk_level = "Low"
        if len(risk_factors) >= 3:
            risk_level = "High"
        elif len(risk_factors) >= 1:
            risk_level = "Medium"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'risk_score': min(len(risk_factors) * 20, 100)
        }
    
    def _interpret_biomarkers(self, texture: Dict, morphology: Dict, intensity: Dict) -> List[str]:
        """تفسير سريري للمؤشرات الحيوية"""
        interpretations = []
        
        # تفسيرات النسيج
        entropy = texture.get('entropy', 0)
        if entropy > 3.0:
            interpretations.append("High tissue entropy suggests structural heterogeneity")
        elif entropy < 1.5:
            interpretations.append("Low tissue entropy indicates uniform tissue structure")
        
        # تفسيرات الشكل
        edge_density = morphology.get('edge_density', 0)
        if edge_density > 0.25:
            interpretations.append("Elevated edge density may indicate complex tissue boundaries")
        
        # تفسيرات الشدة
        intensity_std = intensity.get('std_intensity', 0)
        if intensity_std > 70:
            interpretations.append("High intensity variability could suggest mixed tissue composition")
        
        return interpretations
    
    # الدوال الافتراضية
    def _get_default_texture_features(self) -> Dict:
        return {key: 0.0 for key in ['contrast', 'homogeneity', 'energy', 'correlation', 
                                    'entropy', 'variance', 'smoothness', 'uniformity', 'dissimilarity']}
    
    def _get_default_morphological_features(self) -> Dict:
        return {key: 0.0 for key in ['region_count', 'total_area', 'largest_area', 'perimeter',
                                   'eccentricity', 'solidity', 'extent', 'equivalent_diameter',
                                   'edge_density', 'circularity', 'compactness']}
    
    def _get_default_spatial_features(self) -> Dict:
        return {key: 0.0 for key in ['spatial_uniformity', 'max_quarter_contrast', 
                                   'horizontal_symmetry', 'vertical_symmetry']}

# دالة مساعدة للعثور على القمم
def find_peaks(x, height=None, distance=None):
    """بحث مبسط عن القمم (بديل عن scipy.signal.find_peaks)"""
    peaks = []
    properties = {'peak_heights': []}
    
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if height is None or x[i] > height:
                peaks.append(i)
                properties['peak_heights'].append(x[i])
    
    return np.array(peaks), properties

# التهيئة العالمية
@st.cache_resource
def load_biomarker_analyzer():
    """تحميل محلل المؤشرات الحيوية مع التخزين المؤقت"""
    return QuantitativeBiomarkerAnalyzer()

if __name__ == "__main__":
    analyzer = load_biomarker_analyzer()
    st.success("✅ Advanced Biomarker Analyzer is ready!")
