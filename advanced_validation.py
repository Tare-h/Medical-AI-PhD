"""
Advanced Validation System for Medical AI
Cross-validation and statistical testing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
from scipy import stats
import logging
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedMedicalValidator:
    """
    Advanced validation system with statistical testing for medical AI models
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MedicalAI.Validator")
        self.results = {}
        
    def stratified_cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                                  cv_folds: int = 10, model_name: str = '') -> Dict[str, Any]:
        """Comprehensive stratified cross-validation"""
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted', 
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_ovr_weighted'
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = cross_validate(
            model, X, y, 
            cv=cv, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        results_summary = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results_summary[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_scores': test_scores,
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'train_scores': train_scores
            }
        
        self.results[model_name] = results_summary
        
        self.logger.info(f"📊 {model_name} CV Results:")
        self.logger.info(f"   Accuracy:  {results_summary['accuracy']['test_mean']:.4f} ± {results_summary['accuracy']['test_std']:.4f}")
        self.logger.info(f"   F1 Score:  {results_summary['f1']['test_mean']:.4f} ± {results_summary['f1']['test_std']:.4f}")
        self.logger.info(f"   ROC AUC:   {results_summary['roc_auc']['test_mean']:.4f} ± {results_summary['roc_auc']['test_std']:.4f}")
        
        return results_summary
    
    def statistical_significance_test(self, model1_scores: np.ndarray, 
                                    model2_scores: np.ndarray, 
                                    model1_name: str, 
                                    model2_name: str) -> Dict[str, Any]:
        """Perform statistical significance testing between two models"""
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(model1_scores) - np.mean(model2_scores)
        pooled_std = np.sqrt((np.std(model1_scores)**2 + np.std(model2_scores)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
        results = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohens_d,
            'mean_difference': mean_diff,
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores)
        }
        
        significance = "SIGNIFICANT" if results['significant'] else "NOT SIGNIFICANT"
        self.logger.info(f"📈 Statistical test: {model1_name} vs {model2_name}")
        self.logger.info(f"   p-value: {p_value:.6f} ({significance})")
        self.logger.info(f"   Effect size: {cohens_d:.3f}")
        
        return results
    
    def compare_multiple_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Compare multiple models with statistical testing"""
        
        comparison_results = []
        all_scores = {}
        
        # Perform CV for each model
        for name, model in models.items():
            cv_results = self.stratified_cross_validation(model, X, y, model_name=name)
            all_scores[name] = cv_results['f1']['test_scores']
            
            comparison_results.append({
                'Model': name,
                'Accuracy_Mean': cv_results['accuracy']['test_mean'],
                'Accuracy_Std': cv_results['accuracy']['test_std'],
                'F1_Mean': cv_results['f1']['test_mean'],
                'F1_Std': cv_results['f1']['test_std'],
                'ROC_AUC_Mean': cv_results['roc_auc']['test_mean'],
                'ROC_AUC_Std': cv_results['roc_auc']['test_std']
            })
        
        # Perform pairwise statistical tests
        model_names = list(models.keys())
        statistical_tests = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                test_results = self.statistical_significance_test(
                    all_scores[model1], all_scores[model2], model1, model2
                )
                
                statistical_tests.append({
                    'Comparison': f"{model1} vs {model2}",
                    'P_Value': test_results['p_value'],
                    'Significant': test_results['significant'],
                    'Effect_Size': test_results['effect_size'],
                    'Mean_Difference': test_results['mean_difference']
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        stats_df = pd.DataFrame(statistical_tests)
        
        # Save results
        comparison_df.to_csv('results/model_comparison_statistical.csv', index=False)
        stats_df.to_csv('results/statistical_tests.csv', index=False)
        
        return comparison_df, stats_df
    
    def plot_cv_results(self, results_dict: Dict[str, Any]):
        """Plot cross-validation results"""
        
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'f1', 'roc_auc']
        models = list(results_dict.keys())
        
        for idx, metric in enumerate(metrics):
            plt.subplot(2, 2, idx + 1)
            
            means = [results_dict[model][metric]['test_mean'] for model in models]
            stds = [results_dict[model][metric]['test_std'] for model in models]
            
            bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7,
                         color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            
            plt.title(f'{metric.upper()} - Cross Validation', fontweight='bold')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/cross_validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def learning_curve_analysis(self, model, X: np.ndarray, y: np.ndarray, 
                              model_name: str = ''):
        """Generate learning curves for model analysis"""
        
        from sklearn.model_selection import learning_curve
        import numpy as np
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, 
            train_sizes=train_sizes,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        
        plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", 
                label="Training score", linewidth=2)
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g",
                label="Cross-validation score", linewidth=2)
        
        plt.title(f'Learning Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/learning_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"📚 Learning curve generated for {model_name}")
