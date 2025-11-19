"""
Medical AI PhD - Phase 5: Advanced Model Building and Evaluation
Main execution script for comprehensive model development
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import MedicalDataProcessor
from model_factory import MedicalModelFactory
from model_evaluator import MedicalModelEvaluator
from config import DATA_CONFIG
from utils import setup_logging

def main():
    """Main execution function for Phase 5"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 STARTING PHASE 5: Advanced Model Building and Evaluation")
    
    try:
        # Step 1: Data Preparation
        logger.info("📊 Step 1: Loading and preprocessing data...")
        processor = MedicalDataProcessor()
        data = processor.load_and_validate_data()
        
        # Data summary
        summary = processor.get_data_summary(data)
        logger.info(f"   Data Summary: {summary['total_samples']} samples, "
                   f"{len(summary['feature_columns'])} features")
        logger.info(f"   Target Distribution: {summary['target_distribution']}")
        
        # Preprocess data
        X, y = processor.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        logger.info(f"   Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Step 2: Model Factory Initialization
        logger.info("🏭 Step 2: Initializing model factory...")
        model_factory = MedicalModelFactory()
        
        # Create traditional ML models
        model_factory.create_sklearn_models()
        
        # Create deep learning model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model_factory.create_deep_learning_model(input_dim, num_classes)
        
        logger.info(f"   Created {len(model_factory.models)} models for comparison")
        
        # Step 3: Model Training
        logger.info("🎯 Step 3: Training all models...")
        model_factory.train_models(X_train, y_train, X_val, y_val)
        
        logger.info(f"   Successfully trained {len(model_factory.results)} models")
        
        # Step 4: Model Evaluation
        logger.info("📈 Step 4: Evaluating models...")
        evaluation_results = model_factory.evaluate_models(X_test, y_test)
        
        # Step 5: Get Best Model
        logger.info("🏆 Step 5: Identifying best model...")
        best_name, best_model, best_metrics = model_factory.get_best_model()
        
        logger.info(f"   Best Model: {best_name}")
        logger.info(f"   Best Metrics - Accuracy: {best_metrics['accuracy']:.4f}, "
                   f"F1: {best_metrics['f1']:.4f}, AUC: {best_metrics['roc_auc']:.4f}")
        
        # Step 6: Comprehensive Evaluation and Visualization
        logger.info("📊 Step 6: Generating comprehensive evaluations...")
        evaluator = MedicalModelEvaluator()
        
        # Get predictions from best model
        if hasattr(best_model, 'predict_proba'):
            y_pred = best_model.predict(X_test)
            y_scores = best_model.predict_proba(X_test)
        else:
            y_pred = best_model.predict(X_test)
            y_scores = best_model.predict(X_test)
        
        class_names = processor.get_target_names()
        
        # Generate all visualizations
        evaluator.plot_confusion_matrix(y_test, y_pred, class_names, best_name)
        evaluator.plot_roc_curves(y_test, y_scores, class_names, best_name)
        
        # Feature importance for interpretable models
        if hasattr(best_model, 'feature_importances_'):
            evaluator.plot_feature_importance(
                best_model, processor.get_feature_names(), best_name
            )
        
        # Training history for deep learning
        if best_name == 'deep_learning' and model_factory.results[best_name]['history']:
            evaluator.plot_training_history(
                model_factory.results[best_name]['history'], best_name
            )
        
        # Model comparison
        comparison_df = model_factory.get_model_comparison_dataframe()
        evaluator.plot_model_comparison(comparison_df)
        
        # Comprehensive report
        evaluator.generate_comprehensive_report(
            y_test, y_pred, y_scores, class_names, best_name
        )
        
        # Step 7: Save Models and Results
        logger.info("💾 Step 7: Saving models and results...")
        model_factory.save_models()
        
        # Save comparison results
        comparison_df.to_csv('results/model_comparison_results.csv', index=False)
        
        # Generate final report
        training_report = model_factory.generate_training_report()
        logger.info(f"   Training Report: {training_report['total_models_trained']} "
                   f"models trained, {training_report['total_models_evaluated']} evaluated")
        
        # Step 8: Final Summary
        logger.info("✅ PHASE 5 COMPLETED SUCCESSFULLY!")
        logger.info("📋 SUMMARY:")
        logger.info(f"   • Models Trained: {len(model_factory.results)}")
        logger.info(f"   • Best Model: {best_name}")
        logger.info(f"   • Best Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"   • Best F1 Score: {best_metrics['f1']:.4f}")
        logger.info(f"   • Results Saved: results/ folder")
        logger.info(f"   • Models Saved: saved_models/ folder")
        
        print("\n" + "="*60)
        print("🎉 PHASE 5 COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"🏆 Best Model: {best_name}")
        print(f"📊 Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"🎯 F1 Score: {best_metrics['f1']:.4f}")
        print(f"📈 ROC AUC: {best_metrics['roc_auc']:.4f}")
        print(f"💾 Results saved to: results/ folder")
        print(f"💾 Models saved to: saved_models/ folder")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ PHASE 5 FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
