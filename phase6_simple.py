"""
Medical AI PhD - Phase 6: Simplified Hyperparameter Tuning
Basic model optimization without complex dependencies
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import joblib
import json

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase6_simple.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MedicalAI")

class SimpleTuner:
    """Simple hyperparameter tuner for medical models"""
    
    def __init__(self):
        self.best_params = {}
    
    def tune_random_forest(self, X, y):
        """Tune Random Forest parameters"""
        logger.info("🔄 Tuning Random Forest...")
        
        best_score = 0
        best_params = {}
        
        # Simple grid search
        for n_estimators in [100, 200, 300]:
            for max_depth in [10, 15, 20, None]:
                for min_samples_split in [2, 5, 10]:
                    
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                    mean_score = np.mean(scores)
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split
                        }
        
        self.best_params['random_forest'] = best_params
        logger.info(f"✅ Random Forest tuned: F1 = {best_score:.4f}")
        return best_params
    
    def tune_gradient_boosting(self, X, y):
        """Tune Gradient Boosting parameters"""
        logger.info("🔄 Tuning Gradient Boosting...")
        
        best_score = 0
        best_params = {}
        
        for n_estimators in [50, 100, 150]:
            for learning_rate in [0.05, 0.1, 0.2]:
                for max_depth in [3, 4, 5]:
                    
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42
                    )
                    
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
                    mean_score = np.mean(scores)
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth
                        }
        
        self.best_params['gradient_boosting'] = best_params
        logger.info(f"✅ Gradient Boosting tuned: F1 = {best_score:.4f}")
        return best_params
    
    def tune_logistic_regression(self, X, y):
        """Tune Logistic Regression parameters"""
        logger.info("🔄 Tuning Logistic Regression...")
        
        best_score = 0
        best_params = {}
        
        for C in [0.1, 1.0, 10.0]:
            for solver in ['lbfgs', 'liblinear']:
                
                model = LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=1000,
                    random_state=42
                )
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'C': C,
                        'solver': solver
                    }
        
        self.best_params['logistic_regression'] = best_params
        logger.info(f"✅ Logistic Regression tuned: F1 = {best_score:.4f}")
        return best_params

def load_and_preprocess_data():
    """Load and preprocess medical data"""
    try:
        from data_processing import MedicalDataProcessor
        from config import DATA_CONFIG
        
        processor = MedicalDataProcessor()
        data = processor.load_and_validate_data()
        X, y = processor.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        # Combine train and validation for tuning
        X_opt = np.vstack([X_train, X_val])
        y_opt = np.concatenate([y_train, y_val])
        
        return X_opt, y_opt, X_test, y_test, processor
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        # Create sample data for testing
        logger.info("📝 Creating sample data...")
        from utils import create_sample_data
        create_sample_data("data/medical_data.csv")
        
        from data_processing import MedicalDataProcessor
        from config import DATA_CONFIG
        
        processor = MedicalDataProcessor()
        data = processor.load_and_validate_data()
        X, y = processor.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        X_opt = np.vstack([X_train, X_val])
        y_opt = np.concatenate([y_train, y_val])
        
        return X_opt, y_opt, X_test, y_test, processor

def main():
    """Main execution function"""
    logger.info("🚀 STARTING PHASE 6: Simplified Hyperparameter Tuning")
    
    try:
        # Step 1: Load Data
        logger.info("📊 Step 1: Loading data...")
        X_opt, y_opt, X_test, y_test, processor = load_and_preprocess_data()
        logger.info(f"   Data shapes - Optimization: {X_opt.shape}, Test: {X_test.shape}")
        
        # Step 2: Hyperparameter Tuning
        logger.info("🎯 Step 2: Tuning models...")
        tuner = SimpleTuner()
        
        # Tune each model
        rf_params = tuner.tune_random_forest(X_opt, y_opt)
        gb_params = tuner.tune_gradient_boosting(X_opt, y_opt)
        lr_params = tuner.tune_logistic_regression(X_opt, y_opt)
        
        # Step 3: Create Optimized Models
        logger.info("🏭 Step 3: Creating optimized models...")
        optimized_models = {
            'random_forest_opt': RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1),
            'gradient_boosting_opt': GradientBoostingClassifier(**gb_params, random_state=42),
            'logistic_regression_opt': LogisticRegression(**lr_params, random_state=42)
        }
        
        # Step 4: Train Final Models
        logger.info("🎓 Step 4: Training optimized models...")
        for name, model in optimized_models.items():
            model.fit(X_opt, y_opt)
            logger.info(f"   ✅ Trained: {name}")
        
        # Step 5: Evaluate on Test Set
        logger.info("🧪 Step 5: Evaluating on test set...")
        test_results = {}
        
        for name, model in optimized_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            test_results[name] = {
                'accuracy': accuracy,
                'f1': f1
            }
            
            logger.info(f"   {name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        # Step 6: Compare with Baseline
        logger.info("📊 Step 6: Comparing with baseline...")
        baseline_models = {
            'rf_baseline': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb_baseline': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr_baseline': LogisticRegression(random_state=42)
        }
        
        baseline_results = {}
        for name, model in baseline_models.items():
            model.fit(X_opt, y_opt)
            y_pred = model.predict(X_test)
            baseline_results[name] = f1_score(y_test, y_pred, average='weighted')
            logger.info(f"   {name}: F1 = {baseline_results[name]:.4f}")
        
        # Find best models
        best_baseline_f1 = max(baseline_results.values())
        best_optimized_name = max(test_results.keys(), key=lambda x: test_results[x]['f1'])
        best_optimized_f1 = test_results[best_optimized_name]['f1']
        
        improvement = ((best_optimized_f1 - best_baseline_f1) / best_baseline_f1) * 100
        
        # Step 7: Save Results
        logger.info("💾 Step 7: Saving results...")
        
        # Save optimized models
        os.makedirs('saved_models', exist_ok=True)
        for name, model in optimized_models.items():
            joblib.dump(model, f'saved_models/{name}_simple.pkl')
        
        # Save best model
        best_model = optimized_models[best_optimized_name]
        joblib.dump(best_model, 'saved_models/best_optimized_simple.pkl')
        
        # Save results summary
        results_summary = {
            'best_optimized_model': best_optimized_name,
            'best_optimized_f1': best_optimized_f1,
            'best_baseline_f1': best_baseline_f1,
            'improvement_percentage': improvement,
            'tuned_parameters': tuner.best_params,
            'test_results': test_results
        }
        
        with open('results/phase6_simple_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Step 8: Final Report
        logger.info("✅ PHASE 6 COMPLETED SUCCESSFULLY!")
        logger.info("📋 FINAL SUMMARY:")
        logger.info(f"   • Best Optimized Model: {best_optimized_name}")
        logger.info(f"   • Test F1 Score: {best_optimized_f1:.4f}")
        logger.info(f"   • Baseline F1 Score: {best_baseline_f1:.4f}")
        logger.info(f"   • Improvement: {improvement:+.2f}%")
        logger.info(f"   • Models Saved: saved_models/")
        logger.info(f"   • Results Saved: results/")
        
        print("\n" + "="*60)
        print("🎉 PHASE 6: SIMPLIFIED TUNING COMPLETED!")
        print("="*60)
        print(f"🏆 Best Model: {best_optimized_name}")
        print(f"🎯 Test F1 Score: {best_optimized_f1:.4f}")
        print(f"📈 Improvement: {improvement:+.2f}%")
        print(f"💾 Models: saved_models/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ PHASE 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
