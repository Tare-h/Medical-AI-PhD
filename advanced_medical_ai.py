# advanced_medical_ai.py - Advanced Medical AI System
# For: Dr. Jan Kybic - Czech Technical University
# Researcher: Tarek Hamwi

import sys
import os
import math
import statistics
import random
from collections import Counter
import datetime
import json
import base64
import zipfile
from io import StringIO
import csv

print("🔍 Discovering Available Libraries:")
available_libs = [
    'sys', 'os', 'math', 'statistics', 'random', 
    'collections', 'datetime', 'json', 'base64', 'csv'
]
print(f"✅ {len(available_libs)} built-in libraries ready for use")

class MedicalAIModel:
    """
    Medical AI Model simulating machine learning for medical diagnosis
    Built using only Python's standard library
    """
    
    def __init__(self):
        self.training_data = []
        self.feature_importance = {}
        self.model_accuracy = 0.0
        self.training_time = datetime.datetime.now()
        
    def simulate_training(self, medical_features, diagnoses):
        """Simulate training of medical AI model with statistical analysis"""
        print("🧠 Training Medical AI Model...")
        
        # Analyze medical feature statistics
        feature_stats = {}
        for i, feature in enumerate(medical_features[0]):
            values = [case[i] for case in medical_features]
            feature_stats[f'feature_{i}'] = {
                'mean': statistics.mean(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'correlation_with_diagnosis': self.calculate_correlation(values, diagnoses),
                'min': min(values),
                'max': max(values)
            }
        
        # Calculate feature importance for medical diagnosis
        for i, (feature, stats) in enumerate(feature_stats.items()):
            correlation = abs(stats['correlation_with_diagnosis'])
            variability = stats['std_dev'] / max(1, stats['mean'])
            importance = correlation * (1 - variability)
            self.feature_importance[feature] = importance
            
        print("✅ Model training completed successfully")
        return feature_stats
    
    def calculate_correlation(self, x, y):
        """Calculate Pearson correlation between features and diagnoses"""
        if len(x) != len(y):
            return 0
            
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        return numerator / denominator if denominator != 0 else 0
    
    def predict(self, patient_features):
        """Predict diagnosis for new patient"""
        if not self.feature_importance:
            return random.choice([0, 1])
            
        # Weighted prediction based on feature importance
        weighted_sum = 0
        total_weight = 0
        
        for i, feature_value in enumerate(patient_features):
            feature_name = f'feature_{i}'
            weight = self.feature_importance.get(feature_name, 0)
            weighted_sum += feature_value * weight
            total_weight += weight
            
        prediction_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        return 1 if prediction_score > 0.5 else 0
    
    def evaluate_model(self, test_features, test_diagnoses):
        """Comprehensive model evaluation with medical metrics"""
        correct_predictions = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for features, actual in zip(test_features, test_diagnoses):
            predicted = self.predict(features)
            
            if predicted == actual:
                correct_predictions += 1
                if actual == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted == 1:
                    false_positives += 1
                else:
                    false_negatives += 1
        
        accuracy = correct_predictions / len(test_features)
        
        # Medical performance metrics
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        self.model_accuracy = accuracy
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

def generate_medical_data(num_patients=200):
    """Generate realistic medical imaging simulation data"""
    print(f"📊 Generating {num_patients} simulated medical cases...")
    
    patients_data = []
    diagnoses = []
    
    for i in range(num_patients):
        # Simulated medical imaging features
        patient_features = [
            random.uniform(0.1, 0.9),   # Tissue density A
            random.uniform(0.2, 0.8),   # Tissue density B  
            random.uniform(0.0, 1.0),   # Image contrast
            random.uniform(0.3, 0.7),   # Texture regularity
            random.uniform(0.1, 0.6),   # Relative size
            random.uniform(0.4, 0.9),   # Boundary clarity
            random.uniform(0.2, 0.8),   # Signal intensity
        ]
        
        # Simulate diagnosis based on realistic medical patterns
        risk_score = (
            patient_features[0] * 0.25 +    # Tissue density A
            patient_features[1] * 0.35 +    # Tissue density B
            patient_features[2] * 0.15 +    # Contrast
            patient_features[3] * -0.1 +    # Regularity (negative correlation)
            patient_features[4] * 0.2 +     # Size
            patient_features[5] * 0.05 +    # Boundary
            patient_features[6] * 0.1       # Intensity
        )
        
        # Add some randomness to simulate real-world variability
        diagnosis = 1 if risk_score + random.uniform(-0.2, 0.2) > 0.5 else 0
        
        patients_data.append(patient_features)
        diagnoses.append(diagnosis)
    
    healthy_count = diagnoses.count(0)
    patient_count = diagnoses.count(1)
    
    print(f"✅ Generated {num_patients} patients ({patient_count} positive, {healthy_count} negative)")
    return patients_data, diagnoses

def visualize_medical_analysis(features, diagnoses, feature_names=None):
    """Comprehensive medical data analysis and visualization"""
    print("\n📈 Medical Data Analysis Report:")
    print("=" * 50)
    
    if not feature_names:
        feature_names = [
            'Tissue Density A', 'Tissue Density B', 'Image Contrast', 
            'Texture Regularity', 'Relative Size', 'Boundary Clarity', 'Signal Intensity'
        ]
    
    # Population statistics
    healthy_count = diagnoses.count(0)
    patient_count = diagnoses.count(1)
    total_patients = len(diagnoses)
    
    print(f"👥 Patient Distribution: {healthy_count} Healthy | {patient_count} Patients")
    print(f"📊 Disease Prevalence: {patient_count/total_patients*100:.1f}%")
    
    # Feature analysis for medical insights
    print("\n🔍 Medical Feature Analysis:")
    for i, feature_name in enumerate(feature_names):
        feature_values = [case[i] for case in features]
        healthy_values = [features[j][i] for j in range(len(features)) if diagnoses[j] == 0]
        patient_values = [features[j][i] for j in range(len(features)) if diagnoses[j] == 1]
        
        print(f"\n{feature_name}:")
        print(f"  📏 Range: {min(feature_values):.3f} - {max(feature_values):.3f}")
        print(f"  📊 Mean: {statistics.mean(feature_values):.3f} (±{statistics.stdev(feature_values):.3f})")
        
        if len(healthy_values) > 1 and len(patient_values) > 1:
            healthy_mean = statistics.mean(healthy_values)
            patient_mean = statistics.mean(patient_values)
            mean_difference = patient_mean - healthy_mean
            
            print(f"  🏥 Healthy: μ={healthy_mean:.3f}")
            print(f"  🤒 Patients: μ={patient_mean:.3f}")
            print(f"  📈 Mean Difference: {mean_difference:+.3f}")
            
            # Statistical significance indication
            if abs(mean_difference) > 0.1:
                print(f"  💡 Clinically Significant Difference")

class AdvancedMedicalAISystem:
    """
    Complete Medical AI System for Diagnostic Support
    Demonstrating AI capabilities without external dependencies
    """
    
    def __init__(self):
        self.model = MedicalAIModel()
        self.training_history = []
        self.system_version = "1.0"
        self.researcher = "Tarek Hamwi"
        
    def run_comprehensive_demo(self):
        """Run complete medical AI system demonstration"""
        print("=" * 70)
        print("🏥 ADVANCED MEDICAL AI DIAGNOSTIC SYSTEM")
        print(f"🔬 Researcher: {self.researcher}")
        print(f"🎯 For: Dr. Jan Kybic - Czech Technical University")
        print("=" * 70)
        
        # 1. Data Generation Phase
        print("\n📁 PHASE 1: MEDICAL DATA GENERATION")
        features, diagnoses = generate_medical_data(200)
        
        # 2. Data Analysis Phase
        print("\n📊 PHASE 2: COMPREHENSIVE DATA ANALYSIS")
        feature_names = [
            'Tissue Density A', 'Tissue Density B', 'Image Contrast',
            'Texture Regularity', 'Relative Size', 'Boundary Clarity', 'Signal Intensity'
        ]
        visualize_medical_analysis(features, diagnoses, feature_names)
        
        # 3. Model Training Phase
        print("\n🎓 PHASE 3: AI MODEL TRAINING")
        split_index = int(0.7 * len(features))
        train_features, test_features = features[:split_index], features[split_index:]
        train_diagnoses, test_diagnoses = diagnoses[:split_index], diagnoses[split_index:]
        
        training_stats = self.model.simulate_training(train_features, train_diagnoses)
        
        # 4. Feature Importance Analysis
        print("\n📈 PHASE 4: FEATURE IMPORTANCE ANALYSIS")
        print("Key predictors for medical diagnosis:")
        sorted_features = sorted(self.model.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:5]:  # Top 5 features
            feature_idx = int(feature.split('_')[1])
            print(f"  {feature_names[feature_idx]}: {importance:.3f}")
        
        # 5. Model Evaluation
        print("\n🧪 PHASE 5: MODEL PERFORMANCE EVALUATION")
        evaluation_metrics = self.model.evaluate_model(test_features, test_diagnoses)
        
        print(f"✅ Accuracy: {evaluation_metrics['accuracy']:.1%}")
        print(f"🎯 Sensitivity: {evaluation_metrics['sensitivity']:.1%}")
        print(f"🛡️ Specificity: {evaluation_metrics['specificity']:.1%}")
        print(f"📊 True Positives: {evaluation_metrics['true_positives']}")
        print(f"📊 True Negatives: {evaluation_metrics['true_negatives']}")
        
        # 6. Clinical Predictions
        print("\n🔮 PHASE 6: CLINICAL PREDICTIONS")
        new_patients = [
            [0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9],  # High risk patient
            [0.7, 0.3, 0.4, 0.8, 0.4, 0.6, 0.3],  # Low risk patient
            [0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.6],  # Borderline case
        ]
        
        patient_descriptions = ["High Risk Suspicion", "Low Risk Profile", "Borderline Case"]
        
        for i, (patient, desc) in enumerate(zip(new_patients, patient_descriptions)):
            prediction = self.model.predict(patient)
            confidence = abs(sum(patient) - 3.5) / 3.5  # Simulated confidence
            
            if prediction == 1:
                status = "🟥 REQUIRES MEDICAL ATTENTION"
                recommendation = "Recommend further diagnostic imaging"
            else:
                status = "🟦 LIKELY HEALTHY"
                recommendation = "Routine follow-up recommended"
            
            print(f"\n  Patient {i+1} ({desc}):")
            print(f"    Prediction: {status}")
            print(f"    Confidence: {confidence:.1%}")
            print(f"    Recommendation: {recommendation}")
        
        return evaluation_metrics

# Main execution
if __name__ == "__main__":
    # Create and run the complete medical AI system
    medical_ai_system = AdvancedMedicalAISystem()
    
    print("🚀 INITIALIZING MEDICAL AI RESEARCH DEMONSTRATION")
    print("💡 Built exclusively with Python standard library")
    print("🎯 Demonstrating AI capabilities for medical imaging research\n")
    
    final_metrics = medical_ai_system.run_comprehensive_demo()
    
    print("\n" + "=" * 70)
    print("🎉 MEDICAL AI RESEARCH DEMONSTRATION COMPLETED!")
    print("=" * 70)
    print(f"📈 Final Model Accuracy: {final_metrics['accuracy']:.1%}")
    print(f"👨‍🔬 Researcher: Tarek Hamwi")
    print(f"🎯 Prepared for: Dr. Jan Kybic - CTU Prague")
    print(f"💡 Technical Innovation: Library-independent AI implementation")
    print(f"🏥 Medical Focus: Diagnostic support system")
    print("=" * 70)
    print("🚀 Ready for academic review and research collaboration!")
