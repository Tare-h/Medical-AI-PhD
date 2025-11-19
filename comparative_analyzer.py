"""
Advanced Comparative Analysis for Medical Images
Comprehensive timeline analysis and progression tracking
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import streamlit as st
import json

class ComparativeAnalyzer:
    """Advanced Comparative Analysis System for Medical Cases"""
    
    def __init__(self, database_manager):
        self.db = database_manager
        self.risk_mapping = {
            'LOW': 0.2,
            'MODERATE': 0.5,
            'HIGH': 0.8,
            'CRITICAL': 0.95
        }
    
    def get_patient_timeline(self, patient_id: str) -> pd.DataFrame:
        """Get comprehensive patient timeline with enhanced data"""
        try:
            history = self.db.get_patient_history(patient_id)
            if history.empty:
                return pd.DataFrame()
            
            # Convert and enhance data
            history['analysis_date'] = pd.to_datetime(history['analysis_date'])
            history['risk_score'] = history['risk_level'].map(self.risk_mapping).fillna(0.5)
            history['quality_score'] = history['quality_score'].astype(float)
            history['confidence_score'] = history['confidence_score'].astype(float)
            
            # Parse clinical findings for severity analysis
            history['findings_count'] = history['clinical_findings'].apply(
                lambda x: len(json.loads(x)) if x else 0
            )
            
            # Calculate progression metrics
            history = self.calculate_progression_metrics(history)
            
            return history.sort_values('analysis_date')
            
        except Exception as e:
            st.error(f"Error getting patient timeline: {e}")
            return pd.DataFrame()
    
    def calculate_progression_metrics(self, timeline: pd.DataFrame) -> pd.DataFrame:
        """Calculate progression metrics for timeline data"""
        if len(timeline) < 2:
            timeline['risk_change'] = 0
            timeline['quality_change'] = 0
            timeline['progression_rate'] = 0
            return timeline
        
        # Calculate changes between consecutive analyses
        timeline = timeline.sort_values('analysis_date')
        timeline['risk_change'] = timeline['risk_score'].diff()
        timeline['quality_change'] = timeline['quality_score'].diff()
        
        # Calculate progression rate (risk change per month)
        timeline['days_between'] = timeline['analysis_date'].diff().dt.days
        timeline['progression_rate'] = timeline['risk_change'] / (timeline['days_between'] / 30.0)
        
        # Fill NaN values
        timeline = timeline.fillna(0)
        
        return timeline
    
    def generate_comprehensive_comparative_report(self, patient_id: str) -> Dict:
        """Generate comprehensive comparative analysis report"""
        timeline = self.get_patient_timeline(patient_id)
        
        if timeline.empty:
            return {'error': 'No historical data available for comparison'}
        
        report = {
            'patient_id': patient_id,
            'total_analyses': len(timeline),
            'time_period': self.calculate_time_period_metrics(timeline),
            'trend_analysis': self.analyze_comprehensive_trends(timeline),
            'risk_progression': self.analyze_risk_progression_detailed(timeline),
            'quality_analysis': self.analyze_quality_trends(timeline),
            'clinical_insights': self.generate_clinical_insights(timeline),
            'predictive_analysis': self.generate_predictive_insights(timeline),
            'recommendations': self.generate_comparative_recommendations(timeline)
        }
        
        return report
    
    def calculate_time_period_metrics(self, timeline: pd.DataFrame) -> Dict:
        """Calculate comprehensive time period metrics"""
        start_date = timeline['analysis_date'].min()
        end_date = timeline['analysis_date'].max()
        duration_days = (end_date - start_date).days
        
        return {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'duration_days': duration_days,
            'duration_months': round(duration_days / 30, 1),
            'analyses_per_month': len(timeline) / max(duration_days / 30, 1),
            'average_interval_days': timeline['days_between'].mean() if 'days_between' in timeline.columns else 0
        }
    
    def analyze_comprehensive_trends(self, timeline: pd.DataFrame) -> Dict:
        """Analyze comprehensive trends across multiple dimensions"""
        if len(timeline) < 2:
            return self.get_default_trend_analysis()
        
        risk_scores = timeline['risk_score'].values
        quality_scores = timeline['quality_score'].values
        confidence_scores = timeline['confidence_score'].values
        
        # Risk trend analysis
        risk_trend = self.analyze_risk_trend(risk_scores)
        
        # Quality trend analysis
        quality_trend = self.analyze_quality_trend(quality_scores)
        
        # Statistical analysis
        statistical_metrics = self.calculate_statistical_metrics(timeline)
        
        return {
            'risk_trend': risk_trend['direction'],
            'risk_trend_strength': risk_trend['strength'],
            'risk_change_percentage': f"{(risk_scores[-1] - risk_scores[0]) * 100:.1f}%",
            'quality_trend': quality_trend['direction'],
            'quality_change_percentage': f"{(quality_scores[-1] - quality_scores[0]) * 100:.1f}%",
            'confidence_trend': 'Improving' if confidence_scores[-1] > confidence_scores[0] else 'Declining',
            'stability_score': self.calculate_stability_index(timeline),
            'volatility': statistical_metrics['volatility'],
            'acceleration': statistical_metrics['acceleration']
        }
    
    def analyze_risk_trend(self, risk_scores: np.ndarray) -> Dict:
        """Analyze risk trend with strength assessment"""
        if len(risk_scores) < 2:
            return {'direction': 'Stable', 'strength': 'None'}
        
        # Calculate trend using linear regression
        x = np.arange(len(risk_scores))
        slope, _ = np.polyfit(x, risk_scores, 1)
        
        # Determine direction and strength
        if abs(slope) < 0.01:
            direction = 'Stable'
            strength = 'None'
        elif slope > 0.05:
            direction = 'Rapidly Increasing'
            strength = 'Strong'
        elif slope > 0.01:
            direction = 'Slowly Increasing'
            strength = 'Weak'
        elif slope < -0.05:
            direction = 'Rapidly Decreasing'
            strength = 'Strong'
        else:
            direction = 'Slowly Decreasing'
            strength = 'Weak'
        
        return {'direction': direction, 'strength': strength}
    
    def analyze_quality_trend(self, quality_scores: np.ndarray) -> Dict:
        """Analyze quality trend"""
        if len(quality_scores) < 2:
            return {'direction': 'Stable', 'variability': 'Low'}
        
        change = quality_scores[-1] - quality_scores[0]
        variability = np.std(quality_scores)
        
        if abs(change) < 0.05:
            direction = 'Stable'
        elif change > 0.1:
            direction = 'Significantly Improving'
        elif change > 0.05:
            direction = 'Slowly Improving'
        elif change < -0.1:
            direction = 'Significantly Declining'
        else:
            direction = 'Slowly Declining'
        
        variability_level = 'High' if variability > 0.15 else 'Low'
        
        return {'direction': direction, 'variability': variability_level}
    
    def calculate_statistical_metrics(self, timeline: pd.DataFrame) -> Dict:
        """Calculate advanced statistical metrics"""
        risk_scores = timeline['risk_score'].values
        
        if len(risk_scores) < 3:
            return {'volatility': 0, 'acceleration': 0}
        
        # Volatility (standard deviation of changes)
        changes = np.diff(risk_scores)
        volatility = np.std(changes) if len(changes) > 0 else 0
        
        # Acceleration (change in trend)
        if len(risk_scores) >= 3:
            first_half = risk_scores[:len(risk_scores)//2]
            second_half = risk_scores[len(risk_scores)//2:]
            accel = np.mean(second_half) - np.mean(first_half)
        else:
            accel = 0
        
        return {
            'volatility': float(volatility),
            'acceleration': float(accel)
        }
    
    def calculate_stability_index(self, timeline: pd.DataFrame) -> float:
        """Calculate stability index (0-1, higher is more stable)"""
        if len(timeline) < 2:
            return 1.0
        
        risk_scores = timeline['risk_score'].values
        
        # Combine multiple stability measures
        volatility = np.std(risk_scores)
        max_change = np.max(np.abs(np.diff(risk_scores)))
        
        # Normalize to 0-1 scale (higher is better)
        volatility_score = max(0, 1 - volatility * 2)
        change_score = max(0, 1 - max_change * 4)
        
        return (volatility_score + change_score) / 2
    
    def analyze_risk_progression_detailed(self, timeline: pd.DataFrame) -> Dict:
        """Analyze detailed risk progression patterns"""
        risk_levels = timeline['risk_level'].value_counts()
        
        if timeline.empty:
            return self.get_default_risk_progression()
        
        current_risk = timeline.iloc[-1]['risk_level']
        risk_transitions = self.analyze_risk_transitions(timeline)
        
        return {
            'current_risk': current_risk,
            'initial_risk': timeline.iloc[0]['risk_level'] if not timeline.empty else 'Unknown',
            'risk_distribution': risk_levels.to_dict(),
            'most_common_risk': risk_levels.idxmax() if not risk_levels.empty else 'Unknown',
            'risk_consistency': self.assess_risk_consistency(timeline),
            'risk_transitions': risk_transitions,
            'escalation_count': len([t for t in risk_transitions if t['change'] == 'Increase']),
            'improvement_count': len([t for t in risk_transitions if t['change'] == 'Decrease'])
        }
    
    def analyze_risk_transitions(self, timeline: pd.DataFrame) -> List[Dict]:
        """Analyze transitions between risk levels"""
        transitions = []
        
        if len(timeline) < 2:
            return transitions
        
        timeline = timeline.sort_values('analysis_date')
        
        for i in range(1, len(timeline)):
            prev_risk = timeline.iloc[i-1]['risk_level']
            curr_risk = timeline.iloc[i]['risk_level']
            
            if prev_risk != curr_risk:
                risk_order = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
                prev_index = risk_order.index(prev_risk)
                curr_index = risk_order.index(curr_risk)
                
                change = 'Increase' if curr_index > prev_index else 'Decrease'
                magnitude = abs(curr_index - prev_index)
                
                transitions.append({
                    'from': prev_risk,
                    'to': curr_risk,
                    'change': change,
                    'magnitude': magnitude,
                    'date': timeline.iloc[i]['analysis_date'].strftime('%Y-%m-%d')
                })
        
        return transitions
    
    def assess_risk_consistency(self, timeline: pd.DataFrame) -> str:
        """Assess consistency of risk levels over time"""
        if len(timeline) <= 1:
            return 'Consistent'
        
        unique_risks = timeline['risk_level'].nunique()
        
        if unique_risks == 1:
            return 'Highly Consistent'
        elif unique_risks == 2:
            return 'Moderately Consistent'
        else:
            return 'Variable'
    
    def analyze_quality_trends(self, timeline: pd.DataFrame) -> Dict:
        """Analyze image quality trends over time"""
        if timeline.empty:
            return {'trend': 'Unknown', 'average_quality': 0, 'consistency': 'Unknown'}
        
        quality_scores = timeline['quality_score'].values
        
        return {
            'trend': self.analyze_quality_trend(quality_scores)['direction'],
            'average_quality': float(np.mean(quality_scores)),
            'best_quality': float(np.max(quality_scores)),
            'worst_quality': float(np.min(quality_scores)),
            'consistency': 'Consistent' if np.std(quality_scores) < 0.1 else 'Variable',
            'quality_improvement': quality_scores[-1] > quality_scores[0] if len(quality_scores) > 1 else False
        }
    
    def generate_clinical_insights(self, timeline: pd.DataFrame) -> List[str]:
        """Generate clinical insights from comparative analysis"""
        insights = []
        
        if len(timeline) < 2:
            insights.append("Single analysis available. Track progression with follow-up scans.")
            return insights
        
        # Risk progression insights
        risk_change = timeline['risk_score'].iloc[-1] - timeline['risk_score'].iloc[0]
        
        if risk_change > 0.3:
            insights.append("🚨 Significant risk escalation detected - Immediate clinical review recommended")
        elif risk_change > 0.15:
            insights.append("⚠️ Moderate risk increase observed - Close monitoring advised")
        elif risk_change < -0.2:
            insights.append("✅ Notable improvement in risk indicators - Continue current management")
        elif risk_change < -0.1:
            insights.append("📈 Mild improvement observed - Maintain follow-up schedule")
        else:
            insights.append("📊 Stable risk profile - Routine monitoring appropriate")
        
        # Quality insights
        quality_change = timeline['quality_score'].iloc[-1] - timeline['quality_score'].iloc[0]
        if quality_change > 0.1:
            insights.append("🖼️ Recent image quality shows improvement")
        elif quality_change < -0.1:
            insights.append("📷 Consider technical review for image quality consistency")
        
        # Frequency insights
        total_analyses = len(timeline)
        if total_analyses >= 5:
            insights.append(f"📈 Comprehensive dataset available ({total_analyses} analyses)")
        
        # Volatility insights
        risk_volatility = np.std(timeline['risk_score'].values)
        if risk_volatility > 0.2:
            insights.append("🔍 High variability in risk scores - investigate contributing factors")
        
        insights.append("💡 Comparative analysis enhances early detection of subtle changes")
        insights.append("🔄 Regular follow-ups improve trend accuracy and predictive value")
        
        return insights
    
    def generate_predictive_insights(self, timeline: pd.DataFrame) -> Dict:
        """Generate predictive insights based on historical trends"""
        if len(timeline) < 3:
            return {'next_risk_prediction': 'Insufficient data', 'confidence': 'Low'}
        
        risk_scores = timeline['risk_score'].values
        
        # Simple linear projection
        x = np.arange(len(risk_scores))
        slope, intercept = np.polyfit(x, risk_scores, 1)
        
        # Predict next risk score
        next_risk = slope * len(risk_scores) + intercept
        next_risk = max(0, min(1, next_risk))  # Clamp to 0-1
        
        # Map to risk level
        risk_levels = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        thresholds = [0.3, 0.6, 0.85]
        
        predicted_level = 'LOW'
        for i, threshold in enumerate(thresholds):
            if next_risk >= threshold:
                predicted_level = risk_levels[i + 1]
        
        # Confidence based on data quality
        confidence = 'High' if len(timeline) >= 5 and np.std(risk_scores) < 0.15 else 'Medium'
        
        return {
            'next_risk_prediction': predicted_level,
            'predicted_score': float(next_risk),
            'confidence': confidence,
            'projected_trend': 'Increasing' if slope > 0.01 else 'Decreasing' if slope < -0.01 else 'Stable',
            'recommended_next_scan': self.recommend_next_scan(timeline)
        }
    
    def recommend_next_scan(self, timeline: pd.DataFrame) -> str:
        """Recommend timing for next scan based on progression"""
        if len(timeline) < 2:
            return "3-6 months for baseline comparison"
        
        latest_risk = timeline['risk_score'].iloc[-1]
        progression_rate = timeline['progression_rate'].iloc[-1] if 'progression_rate' in timeline.columns else 0
        
        if latest_risk >= 0.7 or progression_rate > 0.1:
            return "1-2 months (High risk/rapid progression)"
        elif latest_risk >= 0.5 or progression_rate > 0.05:
            return "3 months (Moderate risk/progression)"
        else:
            return "6-12 months (Stable/Low risk)"
    
    def generate_comparative_recommendations(self, timeline: pd.DataFrame) -> List[str]:
        """Generate clinical recommendations based on comparative analysis"""
        recommendations = []
        
        if len(timeline) < 2:
            recommendations.append("Establish baseline with initial comprehensive analysis")
            recommendations.append("Schedule follow-up in 3-6 months for progression tracking")
            return recommendations
        
        # Risk-based recommendations
        current_risk = timeline['risk_score'].iloc[-1]
        risk_trend = self.analyze_risk_trend(timeline['risk_score'].values)
        
        if current_risk >= 0.8:
            recommendations.append("🚨 Urgent specialist consultation recommended")
            recommendations.append("Consider advanced imaging or biopsy")
        elif current_risk >= 0.6:
            recommendations.append("Increase monitoring frequency to every 1-2 months")
            recommendations.append("Discuss preventive interventions")
        
        if risk_trend['direction'] in ['Rapidly Increasing', 'Slowly Increasing']:
            recommendations.append("Accelerate follow-up schedule based on progression rate")
        
        # Quality recommendations
        avg_quality = timeline['quality_score'].mean()
        if avg_quality < 0.6:
            recommendations.append("Optimize imaging protocols for better quality")
        
        # General recommendations
        recommendations.append("Maintain detailed imaging records for longitudinal comparison")
        recommendations.append("Correlate imaging findings with clinical symptoms")
        
        return recommendations
    
    def get_default_trend_analysis(self) -> Dict:
        """Return default values when insufficient data"""
        return {
            'risk_trend': 'Unknown',
            'risk_trend_strength': 'None',
            'risk_change_percentage': '0%',
            'quality_trend': 'Unknown',
            'quality_change_percentage': '0%',
            'confidence_trend': 'Unknown',
            'stability_score': 1.0,
            'volatility': 0,
            'acceleration': 0
        }
    
    def get_default_risk_progression(self) -> Dict:
        """Return default risk progression values"""
        return {
            'current_risk': 'Unknown',
            'initial_risk': 'Unknown',
            'risk_distribution': {},
            'most_common_risk': 'Unknown',
            'risk_consistency': 'Unknown',
            'risk_transitions': [],
            'escalation_count': 0,
            'improvement_count': 0
        }
    
    def create_comprehensive_timeline_visualization(self, patient_id: str) -> go.Figure:
        """Create comprehensive timeline visualization"""
        timeline = self.get_patient_timeline(patient_id)
        
        if timeline.empty:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Risk Score Progression',
                'Image Quality Trends', 
                'Clinical Findings Count'
            ),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Risk progression plot
        fig.add_trace(
            go.Scatter(
                x=timeline['analysis_date'],
                y=timeline['risk_score'],
                mode='lines+markers+text',
                name='Risk Score',
                line=dict(color='red', width=3),
                marker=dict(size=8, color='red'),
                text=timeline['risk_level'],
                textposition="top center",
                hovertemplate='<b>Date:</b> %{x}<br><b>Risk Score:</b> %{y:.3f}<br><b>Level:</b> %{text}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Quality trends plot
        fig.add_trace(
            go.Scatter(
                x=timeline['analysis_date'],
                y=timeline['quality_score'],
                mode='lines+markers',
                name='Quality Score',
                line=dict(color='blue', width=3),
                marker=dict(size=8, color='blue'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Quality Score:</b> %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Findings count plot
        fig.add_trace(
            go.Bar(
                x=timeline['analysis_date'],
                y=timeline['findings_count'],
                name='Findings Count',
                marker_color='orange',
                opacity=0.7,
                hovertemplate='<b>Date:</b> %{x}<br><b>Findings Count:</b> %{y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Comprehensive Timeline Analysis - Patient {patient_id}",
            showlegend=True,
            template="plotly_white",
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Analysis Date", row=3, col=1)
        fig.update_yaxes(title_text="Risk Score (0-1)", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Quality Score (0-1)", row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Number of Findings", row=3, col=1)
        
        return fig
    
    def create_risk_distribution_chart(self, patient_id: str) -> go.Figure:
        """Create risk distribution pie chart"""
        timeline = self.get_patient_timeline(patient_id)
        
        if timeline.empty:
            return self.create_empty_chart("No data available")
        
        risk_counts = timeline['risk_level'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=.3,
            marker_colors=['green', 'orange', 'red', 'darkred']
        )])
        
        fig.update_layout(
            title_text="Risk Level Distribution",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_quality_trend_chart(self, patient_id: str) -> go.Figure:
        """Create quality trend analysis chart"""
        timeline = self.get_patient_timeline(patient_id)
        
        if timeline.empty:
            return self.create_empty_chart("No quality data available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timeline['analysis_date'],
            y=timeline['quality_score'],
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='blue', width=3)
        ))
        
        # Add trend line
        if len(timeline) > 1:
            z = np.polyfit(range(len(timeline)), timeline['quality_score'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=timeline['analysis_date'],
                y=p(range(len(timeline))),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title_text="Image Quality Trend Analysis",
            xaxis_title="Analysis Date",
            yaxis_title="Quality Score",
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        return fig
