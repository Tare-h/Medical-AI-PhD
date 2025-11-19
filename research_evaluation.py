import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd

class ResearchEvaluator:
    def __init__(self, class_names=['COVID-19', 'Normal', 'Pneumonia']):
        self.class_names = class_names
        self.results = {}
        
    def comprehensive_evaluation(self, y_true, y_pred, y_pred_proba):
        """Comprehensive evaluation for research paper"""
        
        # Classification report
        clf_report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC for multi-class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate average AUC
        roc_auc["macro"] = np.mean(list(roc_auc.values()))
        
        self.results = {
            'classification_report': clf_report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        return self.results
    
    def generate_research_metrics(self):
        """Generate research-grade metrics"""
        if not self.results:
            return "No evaluation results available"
        
        report = self.results['classification_report']
        roc_auc = self.results['roc_auc']
        
        metrics_df = pd.DataFrame({
            'Class': self.class_names + ['Macro Average'],
            'Precision': [report[cls]['precision'] for cls in self.class_names] + [report['macro avg']['precision']],
            'Recall': [report[cls]['recall'] for cls in self.class_names] + [report['macro avg']['recall']],
            'F1-Score': [report[cls]['f1-score'] for cls in self.class_names] + [report['macro avg']['f1-score']],
            'ROC-AUC': [roc_auc[i] for i in range(len(self.class_names))] + [roc_auc['macro']]
        })
        
        return metrics_df
    
    def plot_research_figures(self, save_path='research_paper/figures'):
        """Generate research-quality figures"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Research Evaluation', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: ROC Curves
        plt.figure(figsize=(10, 8))
        for i in range(len(self.class_names)):
            plt.plot(self.results['fpr'][i], self.results['tpr'][i],
                    label=f'{self.class_names[i]} (AUC = {self.results["roc_auc"][i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Multi-class ROC Curves - Research Grade', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Research figures saved to {save_path}")

# Test evaluation system
if __name__ == "__main__":
    # Simulate research data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    y_pred_proba = np.random.dirichlet(np.ones(3), 1000)
    
    evaluator = ResearchEvaluator()
    results = evaluator.comprehensive_evaluation(y_true, y_pred, y_pred_proba)
    
    metrics_df = evaluator.generate_research_metrics()
    print("📊 Research Metrics:")
    print(metrics_df)
    
    evaluator.plot_research_figures()
