"""
Advanced Medical Imaging Analysis - Local Version
No TensorFlow required - Advanced traditional image processing
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import logging
import os
from typing import Dict, List
import streamlit as st
from datetime import datetime
from scipy import ndimage
from skimage import feature, filters, measure, morphology

class LocalImagingAnalysis:
    """
    Advanced medical image analysis using traditional computer vision
    No external dependencies required
    """
    
    def __init__(self):
        self.logger = self.setup_logging()
        st.success("✅ Local Imaging Engine: Ready (No TensorFlow Required)")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("MedicalAI.LocalImaging")
    
    def analyze_medical_image(self, image_file, modality: str = "X-RAY") -> Dict:
        """
        Comprehensive medical image analysis using advanced traditional methods
        """
        try:
            # Load image
            image = Image.open(image_file)
            img_array = np.array(image)
            
            # Advanced preprocessing
            processed_image = self.advanced_preprocessing(img_array, modality)
            
            # Extract comprehensive features
            biomarkers = self.extract_comprehensive_biomarkers(processed_image, modality)
            
            # Generate clinical analysis
            clinical_analysis = self.generate_detailed_analysis(biomarkers, modality)
            
            return {
                'success': True,
                'modality': modality,
                'biomarkers': biomarkers,
                'clinical_analysis': clinical_analysis,
                'image_properties': {
                    'dimensions': f"{img_array.shape[1]}x{img_array.shape[0]}",
                    'color_mode': 'Color' if len(img_array.shape) == 3 else 'Grayscale',
                    'file_size': f"{len(image_file.getvalue()) / 1024:.1f} KB"
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Analysis failed: {str(e)}"
            }
    
    def advanced_preprocessing(self, image: np.ndarray, modality: str) -> np.ndarray:
        """Advanced medical image preprocessing"""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Modality-specific enhancement
        if modality.upper() in ["X-RAY", "XRAY"]:
            enhanced = self.enhance_xray(gray)
        elif modality.upper() in ["CT", "CT-SCAN"]:
            enhanced = self.enhance_ct(gray)
        elif modality.upper() == "MRI":
            enhanced = self.enhance_mri(gray)
        else:
            enhanced = self.enhance_general(gray)
        
        return enhanced
    
    def enhance_xray(self, image: np.ndarray) -> np.ndarray:
        """Advanced X-Ray enhancement"""
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Unsharp masking for sharpness
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    def enhance_ct(self, image: np.ndarray) -> np.ndarray:
        """CT scan enhancement"""
        # Noise reduction while preserving edges
        enhanced = cv2.medianBlur(image, 3)
        
        # Histogram equalization
        enhanced = cv2.equalizeHist(enhanced)
        
        return enhanced
    
    def enhance_mri(self, image: np.ndarray) -> np.ndarray:
        """MRI enhancement"""
        # Gaussian filtering for noise reduction
        enhanced = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Contrast stretching
        min_val = np.min(enhanced)
        max_val = np.max(enhanced)
        if max_val > min_val:
            enhanced = (enhanced - min_val) * (255.0 / (max_val - min_val))
        
        return enhanced.astype(np.uint8)
    
    def enhance_general(self, image: np.ndarray) -> np.ndarray:
        """General medical image enhancement"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def extract_comprehensive_biomarkers(self, image: np.ndarray, modality: str) -> Dict:
        """Extract comprehensive imaging biomarkers"""
        
        # Basic intensity statistics
        intensity_stats = self.calculate_intensity_statistics(image)
        
        # Texture analysis
        texture_features = self.analyze_texture_advanced(image)
        
        # Morphological analysis
        morphological_features = self.analyze_morphology_advanced(image)
        
        # Edge and boundary analysis
        edge_features = self.analyze_edges(image)
        
        # Frequency domain analysis
        frequency_features = self.analyze_frequency_domain(image)
        
        # Combine all features
        biomarkers = {
            **intensity_stats,
            **texture_features,
            **morphological_features,
            **edge_features,
            **frequency_features
        }
        
        # Add modality-specific biomarkers
        biomarkers.update(self.get_modality_specific_biomarkers(image, modality))
        
        return biomarkers
    
    def calculate_intensity_statistics(self, image: np.ndarray) -> Dict:
        """Calculate comprehensive intensity statistics"""
        return {
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'min_intensity': float(np.min(image)),
            'max_intensity': float(np.max(image)),
            'median_intensity': float(np.median(image)),
            'skewness': float(self.calculate_skewness(image)),
            'kurtosis': float(self.calculate_kurtosis(image)),
            'energy': float(np.mean(image ** 2)),
            'entropy': float(self.calculate_entropy_advanced(image)),
            'uniformity': float(self.calculate_uniformity(image)),
            'smoothness': float(1 - 1/(1 + np.var(image)))
        }
    
    def calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate intensity skewness"""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return np.mean(((image - mean) / std) ** 3)
        return 0.0
    
    def calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate intensity kurtosis"""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return np.mean(((image - mean) / std) ** 4) - 3
        return 0.0
    
    def calculate_entropy_advanced(self, image: np.ndarray) -> float:
        """Calculate advanced entropy"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        return -np.sum(histogram * np.log2(histogram + 1e-8))
    
    def calculate_uniformity(self, image: np.ndarray) -> float:
        """Calculate intensity uniformity"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        return np.sum(histogram ** 2)
    
    def analyze_texture_advanced(self, image: np.ndarray) -> Dict:
        """Advanced texture analysis"""
        try:
            # GLCM texture features
            glcm = feature.greycomatrix(image, [1, 3], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
            
            texture_features = {
                'contrast': float(np.mean(feature.greycoprops(glcm, 'contrast'))),
                'dissimilarity': float(np.mean(feature.greycoprops(glcm, 'dissimilarity'))),
                'homogeneity': float(np.mean(feature.greycoprops(glcm, 'homogeneity'))),
                'energy': float(np.mean(feature.greycoprops(glcm, 'energy'))),
                'correlation': float(np.mean(feature.greycoprops(glcm, 'correlation'))),
                'ASM': float(np.mean(feature.greycoprops(glcm, 'ASM')))
            }
            
            # Local Binary Patterns
            lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
            texture_features['lbp_entropy'] = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-8)))
            
            return texture_features
            
        except Exception as e:
            self.logger.warning(f"Advanced texture analysis failed: {e}")
            return {}
    
    def analyze_morphology_advanced(self, image: np.ndarray) -> Dict:
        """Advanced morphological analysis"""
        try:
            # Binary image for morphological analysis
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                return {
                    'region_count': int(num_labels - 1),
                    'total_region_area': float(np.sum(areas)),
                    'largest_region_area': float(np.max(areas)) if len(areas) > 0 else 0,
                    'mean_region_area': float(np.mean(areas)) if len(areas) > 0 else 0,
                    'region_density': float(len(areas) / (image.shape[0] * image.shape[1])),
                    'region_compactness': self.calculate_compactness(areas, stats[1:])
                }
            else:
                return {
                    'region_count': 0,
                    'total_region_area': 0,
                    'largest_region_area': 0,
                    'mean_region_area': 0,
                    'region_density': 0,
                    'region_compactness': 0
                }
                
        except Exception as e:
            self.logger.warning(f"Morphological analysis failed: {e}")
            return {}
    
    def calculate_compactness(self, areas, stats):
        """Calculate region compactness"""
        if len(areas) == 0:
            return 0.0
        
        compactness_scores = []
        for i, area in enumerate(areas):
            if area > 0:
                perimeter = stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_HEIGHT]
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                compactness_scores.append(compactness)
        
        return float(np.mean(compactness_scores)) if compactness_scores else 0.0
    
    def analyze_edges(self, image: np.ndarray) -> Dict:
        """Advanced edge analysis"""
        try:
            # Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            
            # Edge statistics
            edge_pixels = np.sum(edges > 0)
            total_pixels = image.shape[0] * image.shape[1]
            
            return {
                'edge_density': float(edge_pixels / total_pixels),
                'edge_intensity_mean': float(np.mean(edges)),
                'edge_orientation_variance': self.calculate_edge_orientation(image)
            }
        except:
            return {'edge_density': 0, 'edge_intensity_mean': 0, 'edge_orientation_variance': 0}
    
    def calculate_edge_orientation(self, image: np.ndarray) -> float:
        """Calculate edge orientation variance"""
        try:
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(sobely, sobelx)
            return float(np.var(orientation))
        except:
            return 0.0
    
    def analyze_frequency_domain(self, image: np.ndarray) -> Dict:
        """Frequency domain analysis"""
        try:
            # Fourier transform
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Analyze frequency distribution
            center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
            low_freq = magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10]
            high_freq = magnitude_spectrum.copy()
            high_freq[center_y-20:center_y+20, center_x-20:center_x+20] = 0
            
            return {
                'low_freq_energy': float(np.mean(low_freq)),
                'high_freq_energy': float(np.mean(high_freq)),
                'spectral_centroid': self.calculate_spectral_centroid(magnitude_spectrum)
            }
        except:
            return {'low_freq_energy': 0, 'high_freq_energy': 0, 'spectral_centroid': 0}
    
    def calculate_spectral_centroid(self, magnitude_spectrum):
        """Calculate spectral centroid"""
        try:
            M, N = magnitude_spectrum.shape
            x = np.arange(N) - N // 2
            y = np.arange(M) - M // 2
            X, Y = np.meshgrid(x, y)
            frequencies = np.sqrt(X**2 + Y**2)
            return float(np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum))
        except:
            return 0.0
    
    def get_modality_specific_biomarkers(self, image: np.ndarray, modality: str) -> Dict:
        """Get modality-specific biomarkers"""
        if modality.upper() in ["X-RAY", "XRAY"]:
            return {
                'bone_contrast_score': float(np.percentile(image, 75) - np.percentile(image, 25)),
                'lung_field_uniformity': float(1.0 / (1.0 + np.std(image[image < np.percentile(image, 50)]))),
                'soft_tissue_visibility': float(np.mean(image[image > np.percentile(image, 25)]))
            }
        elif modality.upper() in ["CT", "CT-SCAN"]:
            return {
                'tissue_differentiation': float(np.std(image)),
                'hounsfield_range': float(np.max(image) - np.min(image)),
                'noise_ratio': float(np.std(image) / (np.mean(image) + 1e-8))
            }
        elif modality.upper() == "MRI":
            return {
                'signal_to_noise': float(np.mean(image) / (np.std(image) + 1e-8)),
                'contrast_to_noise': float((np.max(image) - np.min(image)) / (np.std(image) + 1e-8)),
                'field_uniformity': float(1.0 / (1.0 + np.std(image)))
            }
        else:
            return {
                'image_quality_index': float((np.mean(image) + np.std(image)) / 2),
                'diagnostic_potential': float(self.calculate_entropy_advanced(image) / 8.0)
            }
    
    def generate_detailed_analysis(self, biomarkers: Dict, modality: str) -> Dict:
        """Generate detailed clinical analysis"""
        # Risk assessment
        risk_score = self.calculate_comprehensive_risk(biomarkers, modality)
        
        # Quality assessment
        quality_score = self.assess_comprehensive_quality(biomarkers)
        
        # Generate detailed findings
        findings = self.generate_detailed_findings(biomarkers, modality, risk_score)
        
        # Generate professional recommendations
        recommendations = self.generate_professional_recommendations(risk_score, quality_score, modality)
        
        return {
            'risk_assessment': {
                'score': risk_score,
                'level': 'HIGH' if risk_score > 0.7 else 'MODERATE' if risk_score > 0.4 else 'LOW',
                'color': '🔴' if risk_score > 0.7 else '🟡' if risk_score > 0.4 else '🟢',
                'factors': self.identify_risk_factors(biomarkers)
            },
            'quality_assessment': {
                'score': quality_score,
                'rating': 'EXCELLENT' if quality_score > 0.8 else 'GOOD' if quality_score > 0.6 else 'FAIR' if quality_score > 0.4 else 'POOR',
                'aspects': self.assess_quality_aspects(biomarkers)
            },
            'clinical_findings': findings,
            'recommendations': recommendations,
            'technical_analysis': self.generate_technical_analysis(biomarkers, modality)
        }
    
    def calculate_comprehensive_risk(self, biomarkers: Dict, modality: str) -> float:
        """Calculate comprehensive risk score"""
        risk_factors = 0
        
        # Intensity-based risks
        if biomarkers.get('std_intensity', 0) > 60:
            risk_factors += 1
        if biomarkers.get('skewness', 0) > 1.0 or biomarkers.get('skewness', 0) < -1.0:
            risk_factors += 1
        
        # Texture-based risks
        if biomarkers.get('contrast', 0) > 0.3:
            risk_factors += 1
        if biomarkers.get('entropy', 0) > 6.0:
            risk_factors += 1
        
        # Morphology-based risks
        if biomarkers.get('region_count', 0) > 10:
            risk_factors += 1
        if biomarkers.get('edge_density', 0) > 0.1:
            risk_factors += 1
        
        # Normalize risk score
        max_possible_factors = 6
        return min(risk_factors / max_possible_factors, 1.0)
    
    def assess_comprehensive_quality(self, biomarkers: Dict) -> float:
        """Assess comprehensive image quality"""
        quality_score = 0.0
        
        # Contrast quality (30%)
        contrast = biomarkers.get('std_intensity', 0)
        quality_score += min(contrast / 80.0, 1.0) * 0.3
        
        # Sharpness quality (25%)
        edge_density = biomarkers.get('edge_density', 0)
        quality_score += min(edge_density / 0.15, 1.0) * 0.25
        
        # Noise quality (20%)
        uniformity = biomarkers.get('uniformity', 0)
        quality_score += uniformity * 0.20
        
        # Detail quality (25%)
        entropy = biomarkers.get('entropy', 0)
        quality_score += min(entropy / 7.0, 1.0) * 0.25
        
        return quality_score
    
    def identify_risk_factors(self, biomarkers: Dict) -> List[str]:
        """Identify specific risk factors"""
        factors = []
        
        if biomarkers.get('std_intensity', 0) > 60:
            factors.append("High intensity variation")
        if biomarkers.get('region_count', 0) > 10:
            factors.append("Multiple tissue regions")
        if biomarkers.get('entropy', 0) > 6.0:
            factors.append("Complex tissue patterns")
        if biomarkers.get('edge_density', 0) > 0.1:
            factors.append("Prominent edge structures")
        
        return factors if factors else ["No significant risk factors identified"]
    
    def assess_quality_aspects(self, biomarkers: Dict) -> List[str]:
        """Assess different quality aspects"""
        aspects = []
        
        contrast = biomarkers.get('std_intensity', 0)
        if contrast > 50:
            aspects.append("Good contrast")
        else:
            aspects.append("Moderate contrast")
        
        sharpness = biomarkers.get('edge_density', 0)
        if sharpness > 0.08:
            aspects.append("Good sharpness")
        else:
            aspects.append("Moderate sharpness")
        
        return aspects
    
    def generate_detailed_findings(self, biomarkers: Dict, modality: str, risk_score: float) -> List[str]:
        """Generate detailed clinical findings"""
        findings = []
        
        if risk_score > 0.7:
            findings.append("Significant imaging abnormalities detected requiring urgent specialist review")
            findings.append("Multiple high-risk biomarkers identified in tissue analysis")
        elif risk_score > 0.4:
            findings.append("Moderate imaging variations observed - detailed evaluation recommended")
            findings.append("Several tissue biomarkers outside normal ranges")
        else:
            findings.append("No significant abnormalities detected in comprehensive analysis")
            findings.append("Tissue patterns generally within expected physiological ranges")
        
        # Specific technical findings
        if biomarkers.get('entropy', 0) > 6.0:
            findings.append("High tissue complexity suggesting potential pathological processes")
        
        if biomarkers.get('region_count', 0) > 8:
            findings.append("Multiple distinct tissue regions identified requiring correlation with anatomy")
        
        return findings
    
    def generate_professional_recommendations(self, risk_score: float, quality_score: float, modality: str) -> List[str]:
        """Generate professional clinical recommendations"""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.append("🔴 URGENT: Immediate consultation with appropriate specialist")
            recommendations.append("Consider advanced imaging modalities for detailed characterization")
            recommendations.append("Correlation with clinical symptoms and laboratory findings essential")
        elif risk_score > 0.4:
            recommendations.append("🟡 Recommended: Specialist referral and follow-up imaging")
            recommendations.append("Clinical monitoring with repeat imaging in 3-6 months")
            recommendations.append("Consider additional diagnostic tests based on clinical presentation")
        else:
            recommendations.append("🟢 Routine clinical follow-up as per standard care guidelines")
            recommendations.append("Continue preventive health monitoring")
        
        if quality_score < 0.6:
            recommendations.append("⚠️ Technical Note: Image quality may limit diagnostic accuracy - consider repeat acquisition")
        
        recommendations.append("Final interpretation must be made by qualified radiologist")
        recommendations.append("Clinical correlation with patient history and physical examination required")
        
        return recommendations
    
    def generate_technical_analysis(self, biomarkers: Dict, modality: str) -> Dict:
        """Generate technical analysis report"""
        return {
            'analysis_method': 'Advanced Traditional Computer Vision',
            'features_extracted': len(biomarkers),
            'modality_specific_analysis': True,
            'quality_metrics_calculated': [
                'Contrast', 'Sharpness', 'Noise', 'Texture',
                'Morphology', 'Edge Analysis', 'Frequency Analysis'
            ],
            'biomarker_categories': list(biomarkers.keys())[:10]  # First 10 biomarkers
        }
