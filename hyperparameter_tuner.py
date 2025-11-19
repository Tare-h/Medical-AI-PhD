"""
Advanced Hyperparameter Tuning System
Medical AI PhD - Phase 6
"""

import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import logging
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

class AdvancedHyperparameterTuner:
    """
    Advanced hyperparameter tuning using Optuna with medical domain focus
    """
    
    def __init__(self, n_trials: int = 100, cv_folds: int = 5):
        self.logger = logging.getLogger("MedicalAI.Tuner")
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.studies = {}
        self.best_params = {}
        
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest for medical data"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
            # Use stratified k-fold for medical data
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.studies['random_forest'] = study
        self.best_params['random_forest'] = study.best_params
        
        self.logger.info(f"🎯 Random Forest optimization completed")
        self.logger.info(f"   Best score: {study.best_value:.4f}")
        self.logger.info(f"   Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize Gradient Boosting for medical data"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            model = GradientBoostingClassifier(**params, random_state=42)
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.studies['gradient_boosting'] = study
        self.best_params['gradient_boosting'] = study.best_params
        
        self.logger.info(f"🎯 Gradient Boosting optimization completed")
        self.logger.info(f"   Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_svm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize SVM for medical data"""
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            
            # Additional parameters for specific kernels
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            
            model = SVC(probability=True, random_state=42, **params)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Less folds for SVM due to computation
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)  # Fewer trials for SVM
        
        self.studies['svm'] = study
        self.best_params['svm'] = study.best_params
        
        self.logger.info(f"🎯 SVM optimization completed")
        self.logger.info(f"   Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize Logistic Regression for medical data"""
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                'penalty': trial.suggest_categorical('penalty', ['l2', 'l1']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
            
            # Adjust solver based on penalty
            if params['penalty'] == 'l1':
                params['solver'] = 'liblinear' if params['solver'] != 'saga' else 'saga'
            
            model = LogisticRegression(random_state=42, **params)
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.studies['logistic_regression'] = study
        self.best_params['logistic_regression'] = study.best_params
        
        self.logger.info(f"🎯 Logistic Regression optimization completed")
        self.logger.info(f"   Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize all major models"""
        self.logger.info("🔄 Starting comprehensive hyperparameter optimization...")
        
        results = {}
        
        # Optimize each model type
        results['random_forest'] = self.optimize_random_forest(X, y)
        results['gradient_boosting'] = self.optimize_gradient_boosting(X, y)
        results['logistic_regression'] = self.optimize_logistic_regression(X, y)
        
        # SVM optimization (optional - can be slow)
        try:
            results['svm'] = self.optimize_svm(X, y)
        except Exception as e:
            self.logger.warning(f"SVM optimization skipped: {e}")
        
        self.logger.info("✅ All model optimizations completed")
        return results
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get comprehensive optimization summary"""
        summary_data = []
        
        for model_name, study in self.studies.items():
            summary_data.append({
                'Model': model_name,
                'Best_Score': study.best_value,
                'Best_Params': study.best_params,
                'Trials_Completed': len(study.trials)
            })
        
        return pd.DataFrame(summary_data)
    
    def save_optimization_results(self, directory: str = 'tuning_results'):
        """Save all optimization results"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save best parameters
        best_params_df = self.get_optimization_summary()
        best_params_df.to_csv(f'{directory}/best_parameters.csv', index=False)
        
        # Save detailed trial data
        for model_name, study in self.studies.items():
            trials_df = study.trials_dataframe()
            trials_df.to_csv(f'{directory}/{model_name}_trials.csv', index=False)
        
        # Save visualization plots
        self.plot_optimization_history(directory)
        
        self.logger.info(f"💾 Optimization results saved to {directory}/")
    
    def plot_optimization_history(self, directory: str):
        """Plot optimization history for all studies"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            sns.set_palette("husl")
            
            n_models = len(self.studies)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, (model_name, study) in enumerate(self.studies.items()):
                if idx < 4:  # Limit to 4 subplots
                    # Get trial values
                    trials = study.trials_dataframe()
                    
                    axes[idx].plot(trials['number'], trials['value'], 'o-', alpha=0.6)
                    axes[idx].axhline(y=study.best_value, color='red', linestyle='--', 
                                    label=f'Best: {study.best_value:.4f}')
                    
                    axes[idx].set_title(f'{model_name.replace("_", " ").title()} Optimization', 
                                      fontweight='bold')
                    axes[idx].set_xlabel('Trial Number')
                    axes[idx].set_ylabel('F1 Score')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{directory}/optimization_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create optimization plots: {e}")
