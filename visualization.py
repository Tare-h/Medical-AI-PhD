"""
Advanced Visualization Engine for Medical AI Analytics
محرك التصورات المتقدم للتحليلات الطبية بالذكاء الاصطناعي
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import colorsys

class MedicalVisualizationEngine:
    """محرك التصورات الطبية المتقدم"""
    
    def __init__(self):
        self.color_palette = {
            'diagnosis': {'normal': '#00cc96', 'suspicious': '#ffa500', 'abnormal': '#ff4b4b'},
            'risk': {'low': '#00cc96', 'medium': '#ffa500', 'high': '#ff4b4b', 'critical': '#dc2626'},
            'modality': {'xray': '#6366f1', 'ct': '#8b5cf6', 'mri': '#ec4899', 'mammography': '#f59e0b'}
        }
        
        self.chart_templates = {
            'clinical': 'plotly_white',
            'research': 'plotly_dark',
            'presentation': 'seaborn'
        }
    
    def create_comprehensive_dashboard(self, analysis_data: Dict, patient_data: Dict) -> go.Figure:
        """إنشاء لوحة تحكم شاملة للنتائج الطبية"""
        try:
            # إنشاء شبكة من المخططات
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    'Biomarker Radar Analysis', 'Clinical Risk Assessment',
                    'Texture Features', 'Morphological Analysis', 
                    'Intensity Distribution', 'Spatial Characteristics',
                    'Confidence Metrics', 'Model Performance', 'Timeline'
                ),
                specs=[
                    [{"type": "radar"}, {"type": "indicator"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "histogram"}, {"type": "box"}],
                    [{"type": "gauge"}, {"type": "heatmap"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            # إضافة جميع المخططات
            self._add_biomarker_radar(fig, analysis_data, 1, 1)
            self._add_risk_gauge(fig, analysis_data, 1, 2)
            self._add_texture_bars(fig, analysis_data, 1, 3)
            self._add_morphology_scatter(fig, analysis_data, 2, 1)
            self._add_intensity_histogram(fig, analysis_data, 2, 2)
            self._add_spatial_boxplot(fig, analysis_data, 2, 3)
            self._add_confidence_gauge(fig, analysis_data, 3, 1)
            self._add_performance_heatmap(fig, analysis_data, 3, 2)
            self._add_timeline_scatter(fig, analysis_data, 3, 3)
            
            # تحديث التخطيط
            fig.update_layout(
                height=1000,
                title_text="Comprehensive Medical AI Analysis Dashboard",
                title_x=0.5,
                showlegend=False,
                template=self.chart_templates['clinical'],
                font=dict(size=10)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Dashboard creation failed: {str(e)}")
            return go.Figure()
    
    def _add_biomarker_radar(self, fig, analysis_data, row, col):
        """إضافة مخطط رادار للمؤشرات الحيوية"""
        biomarkers = analysis_data.get('biomarkers', {})
        
        categories = ['Texture', 'Morphology', 'Intensity', 'Statistics', 'Spatial']
        
        texture = biomarkers.get('texture_analysis', {})
        morphology = biomarkers.get('morphological_analysis', {})
        intensity = biomarkers.get('intensity_analysis', {})
        
        values = [
            self._normalize_value(texture.get('entropy', 0), 0, 5),
            self._normalize_value(morphology.get('edge_density', 0), 0, 0.5),
            self._normalize_value(intensity.get('std_intensity', 0), 0, 100),
            self._normalize_value(biomarkers.get('statistical_analysis', {}).get('entropy', 0), 0, 10),
            self._normalize_value(biomarkers.get('spatial_analysis', {}).get('spatial_uniformity', 0), 0, 1)
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.3)',
            line=dict(color='rgb(99, 102, 241)', width=2),
            name='Biomarker Profile'
        ), row=row, col=col)
        
        fig.update_polars(radialaxis=dict(visible=True, range=[0, 1]), row=row, col=col)
    
    def _add_risk_gauge(self, fig, analysis_data, row, col):
        """إضافة مقياس المخاطر"""
        risk_data = analysis_data.get('biomarkers', {}).get('clinical_risk_assessment', {})
        risk_score = risk_data.get('risk_score', 0)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Clinical Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self._get_risk_color(risk_score)},
                'steps': [
                    {'range': [0, 30], 'color': 'lightgray'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=row, col=col)
    
    def _add_texture_bars(self, fig, analysis_data, row, col):
        """إضافة مخطط أشرطة لتحليل النسيج"""
        texture_data = analysis_data.get('biomarkers', {}).get('texture_analysis', {})
        
        features = ['Entropy', 'Contrast', 'Homogeneity', 'Energy', 'Variance']
        values = [
            texture_data.get('entropy', 0),
            texture_data.get('contrast', 0),
            texture_data.get('homogeneity', 0),
            texture_data.get('energy', 0),
            texture_data.get('variance', 0)
        ]
        
        colors = [self._get_value_color(val, [0, 5]) for val in values]
        
        fig.add_trace(go.Bar(
            x=features,
            y=values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        ), row=row, col=col)
        
        fig.update_xaxes(tickangle=45, row=row, col=col)
    
    def _add_morphology_scatter(self, fig, analysis_data, row, col):
        """إضافة مخطط مبعثر للتحليل الشكلي"""
        morphology_data = analysis_data.get('biomarkers', {}).get('morphological_analysis', {})
        
        features = {
            'Edge Density': morphology_data.get('edge_density', 0),
            'Region Count': morphology_data.get('region_count', 0),
            'Solidity': morphology_data.get('solidity', 0),
            'Circularity': morphology_data.get('circularity', 0)
        }
        
        fig.add_trace(go.Scatter(
            x=list(features.keys()),
            y=list(features.values()),
            mode='markers+lines',
            marker=dict(size=12, color='#8b5cf6'),
            line=dict(color='#8b5cf6', width=2),
            name='Morphology'
        ), row=row, col=col)
    
    def _add_intensity_histogram(self, fig, analysis_data, row, col):
        """إضافة مخطط توزيع الشدة"""
        intensity_data = analysis_data.get('biomarkers', {}).get('intensity_analysis', {})
        
        # محاكاة توزيع الشدة
        mean_intensity = intensity_data.get('mean_intensity', 128)
        std_intensity = intensity_data.get('std_intensity', 25)
        
        intensities = np.random.normal(mean_intensity, std_intensity, 1000)
        
        fig.add_trace(go.Histogram(
            x=intensities,
            nbinsx=50,
            marker_color='#ec4899',
            opacity=0.7,
            name='Intensity Distribution'
        ), row=row, col=col)
    
    def _normalize_value(self, value, min_val, max_val):
        """تطبيع القيمة إلى نطاق [0,1]"""
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    def _get_risk_color(self, risk_score):
        """الحصول على لون المخاطر"""
        if risk_score < 30:
            return self.color_palette['risk']['low']
        elif risk_score < 70:
            return self.color_palette['risk']['medium']
        else:
            return self.color_palette['risk']['high']
    
    def _get_value_color(self, value, range_vals):
        """الحصول على لون بناءً على القيمة"""
        normalized = self._normalize_value(value, range_vals[0], range_vals[1])
        if normalized < 0.3:
            return '#00cc96'  # أخضر
        elif normalized < 0.7:
            return '#ffa500'  # برتقالي
        else:
            return '#ff4b4b'  # أحمر
    
    def _add_spatial_boxplot(self, fig, analysis_data, row, col):
        """إضافة مخطط صندوقي للخصائص المكانية"""
        spatial_data = analysis_data.get('biomarkers', {}).get('spatial_analysis', {})
        
        values = [
            spatial_data.get('spatial_uniformity', 0),
            spatial_data.get('max_quarter_contrast', 0),
            spatial_data.get('horizontal_symmetry', 0),
            spatial_data.get('vertical_symmetry', 0)
        ]
        
        fig.add_trace(go.Box(
            y=values,
            name='Spatial Features',
            marker_color='#f59e0b',
            boxpoints='all'
        ), row=row, col=col)
    
    def _add_confidence_gauge(self, fig, analysis_data, row, col):
        """إضافة مقياس الثقة"""
        ai_diagnosis = analysis_data.get('ai_diagnosis', {})
        confidence = ai_diagnosis.get('confidence', 0) * 100
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AI Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "lightgreen"}
                ]
            }
        ), row=row, col=col)
    
    def _add_performance_heatmap(self, fig, analysis_data, row, col):
        """إضافة خريطة حرارية للأداء"""
        # بيانات نموذجية للأداء
        modalities = ['X-Ray', 'CT', 'MRI', 'Mammography']
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
        
        performance_matrix = np.random.uniform(0.85, 0.98, (4, 4))
        
        fig.add_trace(go.Heatmap(
            z=performance_matrix,
            x=metrics,
            y=modalities,
            colorscale='Viridis',
            showscale=True
        ), row=row, col=col)
    
    def _add_timeline_scatter(self, fig, analysis_data, row, col):
        """إضافة مخطط زمني"""
        # بيانات زمنية نموذجية
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        values = np.random.normal(0.9, 0.05, 10)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8, color='#10b981'),
            name='Performance Trend'
        ), row=row, col=col)

# دالة مساعدة
@st.cache_resource
def get_visualization_engine():
    """الحصول على محرك التصورات مع التخزين المؤقت"""
    return MedicalVisualizationEngine()

if __name__ == "__main__":
    viz_engine = MedicalVisualizationEngine()
    st.success("✅ Advanced Visualization Engine is ready!")
