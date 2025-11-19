"""
Advanced Model Evaluation System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
import itertools

class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization system
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MedicalAI.Evaluator")
        plt.style.use('seaborn-v0_8')
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, title='Confusion Matrix'):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_scores, class_names, title='ROC Curves'):
        """Plot multi-class ROC curves"""
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(class_names[i], roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, title='Feature Importance'):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(title, fontsize=16, fontweight='bold')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_comprehensive_report(self, y_true, y_pred, y_scores, class_names, model_name):
        """Generate comprehensive evaluation report"""
        report = {
            'model_name': model_name,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'metrics': {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        }
        
        # Save report
        import json
        with open(f'results/{model_name}_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        self.logger.info(f"Comprehensive report generated for {model_name}")
        return report
