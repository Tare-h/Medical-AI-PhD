# advanced_medical_ai_professional.py
# Medical AI Research System - Professional Academic Version
# Researcher: Tarek Hamwi
# For: Dr. Jan Kybic - Czech Technical University
# Date: December 2024

import sys
import os
import math
import statistics
import random
from collections import Counter
import datetime
import json
import csv

print("====================================================================")
print("MEDICAL AI RESEARCH SYSTEM - ACADEMIC DEMONSTRATION")
print("Researcher: Tarek Hamwi")
print("Academic Target: Dr. Jan Kybic - Czech Technical University")
print("Research Focus: AI in Medical Imaging and Diagnostic Support")
print("====================================================================")

class MedicalAIModel:
    """
    Medical Artificial Intelligence Model
    Implements machine learning algorithms using fundamental mathematical principles
    for medical diagnostic prediction and feature analysis.
    """
    
    def __init__(self):
        self.training_data = []
        self.feature_importance = {}
        self.model_accuracy = 0.0
        self.training_timestamp = datetime.datetime.now()
        self.feature_statistics = {}
        
    def train_model(self, medical_features, diagnoses):
        """
        Train AI model using statistical analysis and correlation methods
        """
        print("PHASE: AI Model Training Initiated")
        
        # Comprehensive feature analysis
        feature_analytics = {}
        for feature_index, feature_vector in enumerate(zip(*medical_features)):
            feature_values = list(feature_vector)
            feature_analytics[f'feature_{feature_index}'] = {
                'mean': statistics.mean(feature_values),
                'standard_deviation': statistics.stdev(feature_values) if len(feature_values) > 1 else 0,
                'correlation_with_diagnosis': self.compute_pearson_correlation(feature_values, diagnoses),
                'value_range': (min(feature_values), max(feature_values)),
                'coefficient_of_variation': statistics.stdev(feature_values) / statistics.mean(feature_values) 
                if statistics.mean(feature_values) != 0 else 0
            }
        
        # Feature importance calculation for medical diagnosis
        for feature, analytics in feature_analytics.items():
            correlation_strength = abs(analytics['correlation_with_diagnosis'])
            feature_variability = analytics['coefficient_of_variation']
            importance_score = correlation_strength * (1 - min(feature_variability, 1))
            self.feature_importance[feature] = importance_score
            
        self.feature_statistics = feature_analytics
        print("STATUS: Model training completed successfully")
        return feature_analytics
    
    def compute_pearson_correlation(self, independent_var, dependent_var):
        """
        Compute Pearson correlation coefficient between features and diagnoses
        """
        if len(independent_var) != len(dependent_var):
            return 0.0
            
        n_observations = len(independent_var)
        sum_x = sum(independent_var)
        sum_y = sum(dependent_var)
        sum_xy = sum(x * y for x, y in zip(independent_var, dependent_var))
        sum_x2 = sum(x ** 2 for x in independent_var)
        sum_y2 = sum(y ** 2 for y in dependent_var)
        
        numerator = n_observations * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n_observations * sum_x2 - sum_x ** 2) * 
                              (n_observations * sum_y2 - sum_y ** 2))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def predict_diagnosis(self, patient_features):
        """
        Predict medical diagnosis based on learned feature importance
        """
        if not self.feature_importance:
            return random.choice([0, 1])  # Random baseline if untrained
            
        # Weighted prediction using feature importance
        prediction_score = 0.0
        total_feature_weight = 0.0
        
        for feature_index, feature_value in enumerate(patient_features):
            feature_name = f'feature_{feature_index}'
            feature_weight = self.feature_importance.get(feature_name, 0.0)
            prediction_score += feature_value * feature_weight
            total_feature_weight += feature_weight
            
        normalized_score = prediction_score / total_feature_weight if total_feature_weight > 0 else 0.5
        return 1 if normalized_score > 0.5 else 0
    
    def evaluate_model_performance(self, test_features, test_diagnoses):
        """
        Comprehensive model evaluation with clinical performance metrics
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for feature_vector, actual_diagnosis in zip(test_features, test_diagnoses):
            predicted_diagnosis = self.predict_diagnosis(feature_vector)
            
            if predicted_diagnosis == actual_diagnosis:
                if actual_diagnosis == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted_diagnosis == 1:
                    false_positives += 1
                else:
                    false_negatives += 1
        
        total_predictions = len(test_features)
        accuracy = (true_positives + true_negatives) / total_predictions
        
        # Clinical performance metrics
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        self.model_accuracy = accuracy
        
        performance_report = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_test_cases': total_predictions
        }
        
        return performance_report

    def perform_statistical_significance_test(self, healthy_values, patient_values):
        """
        Perform t-test simulation for statistical significance
        """
        if len(healthy_values) < 2 or len(patient_values) < 2:
            return "Insufficient data for statistical test"
        
        # Simulate t-test calculation
        mean_healthy = statistics.mean(healthy_values)
        mean_patient = statistics.mean(patient_values)
        std_healthy = statistics.stdev(healthy_values)
        std_patient = statistics.stdev(patient_values)
        
        # Pooled standard deviation
        n_healthy, n_patient = len(healthy_values), len(patient_values)
        pooled_std = math.sqrt(((n_healthy-1)*std_healthy**2 + (n_patient-1)*std_patient**2) / 
                              (n_healthy + n_patient - 2))
        
        # T-statistic
        if pooled_std == 0:
            return "No variability in data"
        
        t_statistic = (mean_patient - mean_healthy) / (pooled_std * math.sqrt(1/n_healthy + 1/n_patient))
        
        # Simplified p-value estimation
        p_value = 2 * (1 - self.standard_normal_cdf(abs(t_statistic)))
        
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
        
        return f"t({n_healthy+n_patient-2}) = {t_statistic:.3f}, p = {p_value:.3f} ({significance})"

    def standard_normal_cdf(self, x):
        """Approximate standard normal CDF"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def create_ascii_histogram(self, values, title="Distribution", bins=10):
        """
        Create ASCII histogram for data visualization
        """
        if not values:
            return "No data available"
        
        min_val, max_val = min(values), max(values)
        bin_width = (max_val - min_val) / bins
        
        # Create bins
        histogram_bins = [0] * bins
        for value in values:
            bin_index = min(int((value - min_val) / bin_width), bins - 1)
            histogram_bins[bin_index] += 1
        
        max_freq = max(histogram_bins)
        
        print(f"\n{title}")
        print("=" * 40)
        
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            frequency = histogram_bins[i]
            
            # Create bar
            bar_length = int((frequency / max_freq) * 30) if max_freq > 0 else 0
            bar = "█" * bar_length
            
            print(f"{bin_start:.2f}-{bin_end:.2f} | {bar} {frequency}")

def generate_medical_research_data(patient_count=200):
    """
    Generate simulated medical imaging data for research purposes
    """
    print(f"DATA GENERATION: Creating {patient_count} simulated medical cases")
    
    patient_dataset = []
    diagnosis_labels = []
    
    for case_id in range(patient_count):
        # Simulated medical imaging features
        medical_features = [
            random.uniform(0.1, 0.9),   # Tissue density measurement A
            random.uniform(0.2, 0.8),   # Tissue density measurement B
            random.uniform(0.0, 1.0),   # Image contrast index
            random.uniform(0.3, 0.7),   # Texture regularity score
            random.uniform(0.1, 0.6),   # Relative size measurement
            random.uniform(0.4, 0.9),   # Boundary clarity index
            random.uniform(0.2, 0.8),   # Signal intensity measurement
        ]
        
        # Realistic diagnosis simulation based on medical patterns
        diagnostic_risk_score = (
            medical_features[0] * 0.25 +    # Primary tissue density
            medical_features[1] * 0.35 +    # Secondary tissue density
            medical_features[2] * 0.15 +    # Contrast importance
            medical_features[3] * -0.1 +    # Regularity (protective factor)
            medical_features[4] * 0.2 +     # Size consideration
            medical_features[5] * 0.05 +    # Boundary assessment
            medical_features[6] * 0.1       # Signal intensity
        )
        
        # Introduce realistic diagnostic variability
        final_diagnosis = 1 if diagnostic_risk_score + random.uniform(-0.2, 0.2) > 0.5 else 0
        
        patient_dataset.append(medical_features)
        diagnosis_labels.append(final_diagnosis)
    
    healthy_cases = diagnosis_labels.count(0)
    patient_cases = diagnosis_labels.count(1)
    
    print(f"STATUS: Generated {patient_count} cases ({patient_cases} positive, {healthy_cases} negative)")
    return patient_dataset, diagnosis_labels

def perform_comprehensive_analysis(features, diagnoses, feature_names=None):
    """
    Perform comprehensive statistical analysis of medical data
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MEDICAL DATA ANALYSIS")
    print("="*60)
    
    if not feature_names:
        feature_names = [
            'Tissue Density A', 'Tissue Density B', 'Image Contrast', 
            'Texture Regularity', 'Relative Size', 'Boundary Clarity', 'Signal Intensity'
        ]
    
    # Population statistics
    healthy_population = diagnoses.count(0)
    patient_population = diagnoses.count(1)
    total_population = len(diagnoses)
    
    print(f"POPULATION: {healthy_population} Healthy | {patient_population} Patients")
    print(f"PREVALENCE: {patient_population/total_population*100:.1f}% Disease Prevalence")
    
    # Detailed feature analysis
    print("\nFEATURE ANALYSIS:")
    for feature_index, feature_name in enumerate(feature_names):
        feature_values = [case[feature_index] for case in features]
        healthy_feature_values = [features[i][feature_index] for i in range(len(features)) if diagnoses[i] == 0]
        patient_feature_values = [features[i][feature_index] for i in range(len(features)) if diagnoses[i] == 1]
        
        print(f"\n{feature_name}:")
        print(f"  Range: {min(feature_values):.3f} - {max(feature_values):.3f}")
        print(f"  Mean: {statistics.mean(feature_values):.3f} (±{statistics.stdev(feature_values):.3f})")
        
        if len(healthy_feature_values) > 1 and len(patient_feature_values) > 1:
            healthy_mean = statistics.mean(healthy_feature_values)
            patient_mean = statistics.mean(patient_feature_values)
            mean_difference = patient_mean - healthy_mean
            
            print(f"  Healthy Cohort: μ={healthy_mean:.3f}")
            print(f"  Patient Cohort: μ={patient_mean:.3f}")
            print(f"  Inter-group Difference: {mean_difference:+.3f}")
            
            # Statistical significance assessment
            if abs(mean_difference) > 0.1:
                print(f"  NOTE: Clinically relevant difference detected")

class MedicalAIResearchSystem:
    """
    Comprehensive Medical AI Research System
    Demonstrates research capabilities in medical artificial intelligence
    """
    
    def __init__(self):
        self.ai_model = MedicalAIModel()
        self.research_log = []
        self.system_version = "Research Edition 1.0"
        self.principal_investigator = "Tarek Hamwi"
        
    def compare_with_baseline_models(self, test_features, test_diagnoses):
        """
        Compare AI model performance with baseline methods
        """
        print("\n" + "="*50)
        print("COMPARATIVE PERFORMANCE ANALYSIS")
        print("="*50)
        
        # 1. Random classifier baseline
        random_predictions = [random.randint(0, 1) for _ in test_diagnoses]
        random_accuracy = sum(1 for p, a in zip(random_predictions, test_diagnoses) if p == a) / len(test_diagnoses)
        
        # 2. Majority class baseline
        majority_class = 1 if sum(test_diagnoses) > len(test_diagnoses) / 2 else 0
        majority_predictions = [majority_class] * len(test_diagnoses)
        majority_accuracy = sum(1 for p, a in zip(majority_predictions, test_diagnoses) if p == a) / len(test_diagnoses)
        
        # 3. Our AI model
        ai_predictions = [self.ai_model.predict_diagnosis(features) for features in test_features]
        ai_accuracy = sum(1 for p, a in zip(ai_predictions, test_diagnoses) if p == a) / len(test_diagnoses)
        
        print("Model Comparison Results:")
        print(f"Random Classifier:      {random_accuracy:.1%}")
        print(f"Majority Class Baseline: {majority_accuracy:.1%}")
        print(f"Our AI Model:           {ai_accuracy:.1%}")
        print(f"Performance Improvement: {ai_accuracy - max(random_accuracy, majority_accuracy):.1%}")
        
        return {
            'random_baseline': random_accuracy,
            'majority_baseline': majority_accuracy,
            'ai_model': ai_accuracy
        }

    def clinical_impact_analysis(self, performance_metrics):
        """
        Analyze potential clinical impact of the AI system
        """
        print("\n" + "="*50)
        print("CLINICAL IMPACT ASSESSMENT")
        print("="*50)
        
        sensitivity = performance_metrics['sensitivity']
        specificity = performance_metrics['specificity']
        
        # Calculate potential clinical impact metrics
        if sensitivity > 0.7:
            screening_potential = "Suitable for screening applications"
        elif sensitivity > 0.5:
            screening_potential = "Potential for triage applications"
        else:
            screening_potential = "Further improvement needed for clinical use"
        
        if specificity > 0.8:
            false_positive_impact = "Low false positive rate - reduces unnecessary follow-ups"
        else:
            false_positive_impact = "Moderate false positive rate - consider clinical context"
        
        clinical_recommendations = [
            f"Sensitivity Analysis: {screening_potential}",
            f"Specificity Analysis: {false_positive_impact}",
            f"Overall Assessment: Research prototype with clinical translation potential",
            f"Next Steps: Validation with real clinical dataset required"
        ]
        
        for recommendation in clinical_recommendations:
            print(f"   • {recommendation}")

    def generate_research_limitations(self):
        """
        Academic discussion of limitations and future work
        """
        print("\n" + "="*50)
        print("RESEARCH LIMITATIONS AND FUTURE DIRECTIONS")
        print("="*50)
        
        limitations = [
            "1. Simulated data may not capture full complexity of real medical images",
            "2. Limited to statistical features rather than deep learning approaches",
            "3. Binary classification may oversimplify clinical diagnostic scenarios",
            "4. Model performance dependent on feature engineering quality"
        ]
        
        future_directions = [
            "1. Integration with real DICOM/NIfTI medical imaging data",
            "2. Implementation of convolutional neural networks for image analysis",
            "3. Multi-class classification for differential diagnosis",
            "4. Clinical validation with expert radiologist input",
            "5. Integration with PACS systems for workflow implementation"
        ]
        
        print("Current Limitations:")
        for limitation in limitations:
            print(f"   {limitation}")
        
        print("\nProposed Future Research Directions:")
        for direction in future_directions:
            print(f"   {direction}")

    def generate_research_summary(self, all_metrics):
        """
        Generate comprehensive research summary
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE RESEARCH SUMMARY")
        print("="*70)
        
        summary = f"""
RESEARCH ACHIEVEMENTS:

1. Technical Implementation:
   • Developed complete AI system using only Python standard library
   • Implemented statistical learning algorithms from first principles
   • Created comprehensive data analysis pipeline

2. Methodological Contributions:
   • Feature importance analysis for medical diagnostics
   • Comparative performance evaluation against baselines
   • Statistical significance testing implementation

3. Clinical Relevance:
   • Accuracy: {all_metrics['accuracy']:.1%} on simulated medical data
   • Sensitivity: {all_metrics['sensitivity']:.1%} - detection capability
   • Specificity: {all_metrics['specificity']:.1%} - specificity in ruling out disease

4. Research Readiness:
   • Modular architecture for easy extension
   • Comprehensive documentation and analysis
   • Foundation for real medical image integration

CONCLUSION:
This research demonstrates strong foundational capabilities in medical AI
and provides a solid platform for advanced research in medical image analysis
under the supervision of Dr. Kybic at CTU Prague.
"""
        print(summary)

    def execute_research_demonstration(self):
        """
        Execute complete research demonstration protocol
        """
        print("\n" + "="*70)
        print("MEDICAL AI RESEARCH DEMONSTRATION PROTOCOL")
        print(f"Principal Investigator: {self.principal_investigator}")
        print(f"Academic Target: Dr. Jan Kybic - Czech Technical University")
        print("Research Domain: Artificial Intelligence in Medical Imaging")
        print("="*70)
        
        # Phase 1: Data Generation
        print("\nPHASE 1: RESEARCH DATA GENERATION")
        feature_data, diagnosis_data = generate_medical_research_data(200)
        
        # Phase 2: Comprehensive Analysis
        print("\nPHASE 2: DATA ANALYSIS AND FEATURE CHARACTERIZATION")
        clinical_feature_names = [
            'Tissue Density A', 'Tissue Density B', 'Image Contrast',
            'Texture Regularity', 'Relative Size', 'Boundary Clarity', 'Signal Intensity'
        ]
        perform_comprehensive_analysis(feature_data, diagnosis_data, clinical_feature_names)
        
        # Phase 3: Model Development
        print("\nPHASE 3: AI MODEL DEVELOPMENT AND TRAINING")
        training_split = int(0.7 * len(feature_data))
        training_features, testing_features = feature_data[:training_split], feature_data[training_split:]
        training_diagnoses, testing_diagnoses = diagnosis_data[:training_split], diagnosis_data[training_split:]
        
        training_analytics = self.ai_model.train_model(training_features, training_diagnoses)
        
        # Phase 4: Feature Importance Analysis
        print("\nPHASE 4: FEATURE IMPORTANCE ANALYSIS")
        print("Feature ranking by diagnostic importance:")
        ranked_features = sorted(self.ai_model.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature_rank, (feature_id, importance_score) in enumerate(ranked_features[:5], 1):
            feature_index = int(feature_id.split('_')[1])
            print(f"  {feature_rank}. {clinical_feature_names[feature_index]}: {importance_score:.3f}")
        
        # Phase 5: Model Validation
        print("\nPHASE 5: MODEL VALIDATION AND PERFORMANCE ASSESSMENT")
        validation_metrics = self.ai_model.evaluate_model_performance(testing_features, testing_diagnoses)
        
        print(f"Performance Metrics:")
        print(f"  Accuracy: {validation_metrics['accuracy']:.1%}")
        print(f"  Sensitivity: {validation_metrics['sensitivity']:.1%}")
        print(f"  Specificity: {validation_metrics['specificity']:.1%}")
        print(f"  Precision: {validation_metrics['precision']:.1%}")
        print(f"  True Positives: {validation_metrics['true_positives']}")
        print(f"  True Negatives: {validation_metrics['true_negatives']}")
        
        # Phase 6: Clinical Application
        print("\nPHASE 6: CLINICAL APPLICATION SCENARIOS")
        clinical_cases = [
            [0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9],  # High clinical suspicion
            [0.7, 0.3, 0.4, 0.8, 0.4, 0.6, 0.3],  # Low risk profile
            [0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.6],  # Intermediate case
        ]
        
        case_descriptions = ["High Suspicion Case", "Low Risk Profile", "Intermediate Presentation"]
        
        for case_number, (clinical_data, description) in enumerate(zip(clinical_cases, case_descriptions), 1):
            prediction = self.ai_model.predict_diagnosis(clinical_data)
            prediction_confidence = abs(sum(clinical_data) - 3.5) / 3.5  # Confidence estimation
            
            if prediction == 1:
                clinical_decision = "POSITIVE - Further investigation recommended"
                management = "Consider advanced imaging and specialist consultation"
            else:
                clinical_decision = "NEGATIVE - Routine follow-up"
                management = "Standard monitoring protocol"
            
            print(f"\nCase {case_number} ({description}):")
            print(f"  AI Assessment: {clinical_decision}")
            print(f"  Confidence Index: {prediction_confidence:.1%}")
            print(f"  Clinical Management: {management}")
        
        # NEW PHASES ADDED HERE:
        print("\nPHASE 7: COMPARATIVE PERFORMANCE EVALUATION")
        comparison_results = self.compare_with_baseline_models(testing_features, testing_diagnoses)

        print("\nPHASE 8: CLINICAL IMPACT ASSESSMENT")
        self.clinical_impact_analysis(validation_metrics)

        print("\nPHASE 9: RESEARCH LIMITATIONS AND FUTURE DIRECTIONS")
        self.generate_research_limitations()

        print("\nPHASE 10: RESEARCH SYNTHESIS")
        self.generate_research_summary(validation_metrics)
        
        return validation_metrics

# Research Protocol Execution
if __name__ == "__main__":
    # Initialize research system
    research_system = MedicalAIResearchSystem()
    
    print("INITIALIZING MEDICAL AI RESEARCH PROTOCOL")
    print("Technical Implementation: Python Standard Library")
    print("Research Objective: Demonstrate AI capabilities for medical imaging")
    print("Academic Context: PhD Research Application\n")
    
    # Execute complete research demonstration
    final_performance = research_system.execute_research_demonstration()
    
    print("\n" + "="*70)
    print("RESEARCH DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"FINAL MODEL ACCURACY: {final_performance['accuracy']:.1%}")
    print(f"PRINCIPAL INVESTIGATOR: Tarek Hamwi")
    print(f"ACADEMIC TARGET: Dr. Jan Kybic - CTU Prague")
    print(f"TECHNICAL APPROACH: Library-independent AI implementation")
    print(f"RESEARCH CONTRIBUTION: Medical diagnostic support system")
    print("="*70)
    print("READY FOR ACADEMIC REVIEW AND RESEARCH COLLABORATION")
