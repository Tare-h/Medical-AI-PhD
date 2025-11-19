import os
import sys
from data_loader import MedicalDataLoader
from research_models import ResearchMedicalModels
from research_evaluation import ResearchEvaluator
import numpy as np

def main():
    print("🔬 Starting Medical AI Research Project...")
    print("=" * 50)
    
    # Step 1: Data Preparation
    print("📥 Step 1: Preparing Research Data...")
    data_loader = MedicalDataLoader()
    data_loader.create_sample_dataset()
    class_names, img_size = data_loader.load_and_preprocess_data()
    
    # Step 2: Model Development
    print("🧠 Step 2: Building Research Models...")
    research_models = ResearchMedicalModels()
    ensemble_models = research_models.build_ensemble_model()
    
    # Step 3: Model Summary for Research Paper
    print("📊 Step 3: Generating Research Metrics...")
    model_summary = research_models.get_model_summary()
    
    # Step 4: Simulate Research Evaluation
    print("📈 Step 4: Comprehensive Research Evaluation...")
    
    # Simulate research results (in real project, this would be actual training)
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    y_pred_proba = np.random.dirichlet(np.ones(3), 1000)
    
    evaluator = ResearchEvaluator(class_names)
    research_results = evaluator.comprehensive_evaluation(y_true, y_pred, y_pred_proba)
    
    # Generate research metrics
    metrics_df = evaluator.generate_research_metrics()
    
    # Generate research figures
    evaluator.plot_research_figures()
    
    # Step 5: Research Summary
    print("\n🎯 RESEARCH PROJECT SUMMARY:")
    print("=" * 40)
    print(f"• Dataset: COVID-19 Chest X-ray Collection")
    print(f"• Classes: {class_names}")
    print(f"• Models: {len(ensemble_models)} ensemble models")
    print(f"• Evaluation: 1000 simulated samples")
    print(f"• Macro AUC: {research_results['roc_auc']['macro']:.3f}")
    print(f"• Figures: Generated in research_paper/figures/")
    print("\n✅ Research project structure completed!")
    print("📁 Next: Add real medical data and train models")

if __name__ == "__main__":
    main()
