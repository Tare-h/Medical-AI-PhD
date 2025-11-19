"""
Medical AI PhD - Phase 6: Advanced Hyperparameter Tuning and Validation
Comprehensive model optimization and statistical validation
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import MedicalDataProcessor
# استيراد الكلاس المطلوب - تم التصحيح
from model_factory import MedicalModelFactory
from hyperparameter_tuner import AdvancedHyperparameterTuner
from advanced_validation import AdvancedMedicalValidator
from config import DATA_CONFIG
from utils import setup_logging

def main():
    """Main execution function for Phase 6"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 STARTING PHASE 6: Advanced Hyperparameter Tuning and Validation")
    
    try:
        # Step 1: Data Preparation
        logger.info("📊 Step 1: Preparing data for optimization...")
        processor = MedicalDataProcessor()
        data = processor.load_and_validate_data()
        
        # Preprocess data
        X, y = processor.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        # Use combined training + validation for optimization
        X_opt = np.vstack([X_train, X_val])
        y_opt = np.concatenate([y_train, y_val])
        
        logger.info(f"   Optimization data: {X_opt.shape}")
        
        # Step 2: Hyperparameter Optimization
        logger.info("🎯 Step 2: Starting hyperparameter optimization...")
        tuner = AdvancedHyperparameterTuner(n_trials=50, cv_folds=5)
        
        # Optimize all models
        best_params = tuner.optimize_all_models(X_opt, y_opt)
        
        # Save optimization results
        tuner.save_optimization_results()
        
        # Step 3: Create Optimized Models
        logger.info("🏭 Step 3: Creating optimized models...")
        optimized_models = {}
        
        # Random Forest with optimized parameters
        if 'random_forest' in best_params:
            optimized_models['random_forest_opt'] = RandomForestClassifier(
                **best_params['random_forest'],
                random_state=42,
                n_jobs=-1
            )
        
        # Gradient Boosting with optimized parameters
        if 'gradient_boosting' in best_params:
            optimized_models['gradient_boosting_opt'] = GradientBoostingClassifier(
                **best_params['gradient_boosting'],
                random_state=42
            )
        
        # Logistic Regression with optimized parameters
        if 'logistic_regression' in best_params:
            optimized_models['logistic_regression_opt'] = LogisticRegression(
                **best_params['logistic_regression'],
                random_state=42
            )
        
        logger.info(f"   Created {len(optimized_models)} optimized models")
        
        # Step 4: Advanced Validation
        logger.info("📈 Step 4: Performing advanced validation...")
        validator = AdvancedMedicalValidator()
        
        # Compare optimized models with comprehensive validation
        comparison_df, stats_df = validator.compare_multiple_models(optimized_models, X_opt, y_opt)
        
        # Plot validation results
        validator.plot_cv_results(validator.results)
        
        # Step 5: Train Final Optimized Models
        logger.info("🎓 Step 5: Training final optimized models...")
        final_models = {}
        
        for name, model in optimized_models.items():
            model.fit(X_opt, y_opt)
            final_models[name] = model
            logger.info(f"   ✅ Trained: {name}")
        
        # Step 6: Evaluate on Test Set
        logger.info("🧪 Step 6: Final evaluation on test set...")
        test_results = {}
        
        for name, model in final_models.items():
            y_pred = model.predict(X_test)
            
            test_accuracy = np.mean(y_pred == y_test)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            test_results[name] = {
                'accuracy': test_accuracy,
                'f1': test_f1
            }
            
            logger.info(f"   {name}: Accuracy = {test_accuracy:.4f}, F1 = {test_f1:.4f}")
        
        # Step 7: Identify Best Optimized Model
        best_optimized_name = max(test_results.keys(), 
                                key=lambda x: test_results[x]['f1'])
        best_optimized_model = final_models[best_optimized_name]
        best_optimized_metrics = test_results[best_optimized_name]
        
        logger.info(f"🏆 Best optimized model: {best_optimized_name}")
        logger.info(f"   Test Accuracy: {best_optimized_metrics['accuracy']:.4f}")
        logger.info(f"   Test F1: {best_optimized_metrics['f1']:.4f}")
        
        # Step 8: Save Final Results
        logger.info("💾 Step 8: Saving final results...")
        
        # Save optimized models
        import joblib
        for name, model in final_models.items():
            joblib.dump(model, f'saved_models/{name}_final.pkl')
        
        # Save best model separately
        joblib.dump(best_optimized_model, 'saved_models/best_optimized_model.pkl')
        
        # Create final results summary
        final_summary = {
            'best_model': best_optimized_name,
            'test_accuracy': best_optimized_metrics['accuracy'],
            'test_f1': best_optimized_metrics['f1'],
            'optimization_trials': tuner.n_trials,
            'models_optimized': len(optimized_models)
        }
        
        import json
        with open('results/phase6_final_summary.json', 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        # Step 9: Final Report
        logger.info("✅ PHASE 6 COMPLETED SUCCESSFULLY!")
        logger.info("📋 FINAL SUMMARY:")
        logger.info(f"   • Models Optimized: {len(optimized_models)}")
        logger.info(f"   • Best Model: {best_optimized_name}")
        logger.info(f"   • Test Accuracy: {best_optimized_metrics['accuracy']:.4f}")
        logger.info(f"   • Test F1 Score: {best_optimized_metrics['f1']:.4f}")
        logger.info(f"   • Results Saved: results/ folder")
        logger.info(f"   • Models Saved: saved_models/ folder")
        
        print("\n" + "="*70)
        print("🎉 PHASE 6: ADVANCED TUNING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"🏆 Best Optimized Model: {best_optimized_name}")
        print(f"📊 Test Accuracy: {best_optimized_metrics['accuracy']:.4f}")
        print(f"🎯 Test F1 Score: {best_optimized_metrics['f1']:.4f}")
        print(f"💾 Optimization results: tuning_results/")
        print(f"💾 Final models: saved_models/")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ PHASE 6 FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
