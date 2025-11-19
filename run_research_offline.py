import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

print("🔬 Starting Medical AI Research Project (Offline Version)...")
print("=" * 50)

# Step 1: Data Preparation
print("📥 Step 1: Preparing Research Data...")

class MedicalDataLoader:
    def create_sample_dataset(self):
        os.makedirs('datasets/images', exist_ok=True)
        os.makedirs('datasets/train/COVID-19', exist_ok=True)
        os.makedirs('datasets/train/Normal', exist_ok=True) 
        os.makedirs('datasets/train/Pneumonia', exist_ok=True)
        os.makedirs('datasets/test/COVID-19', exist_ok=True)
        os.makedirs('datasets/test/Normal', exist_ok=True)
        os.makedirs('datasets/test/Pneumonia', exist_ok=True)
        print("✅ Research dataset structure created")
        
    def load_and_preprocess_data(self, img_size=(224, 224)):
        class_names = ['COVID-19', 'Normal', 'Pneumonia']
        print(f"📊 Research Dataset Summary:")
        print(f"   - Classes: {class_names}")
        print(f"   - Image size: {img_size}")
        return class_names, img_size

data_loader = MedicalDataLoader()
data_loader.create_sample_dataset()
class_names, img_size = data_loader.load_and_preprocess_data()

# Step 2: Model Development (Offline - No Internet Required)
print("🧠 Step 2: Building Research Models (Offline)...")

class ResearchMedicalModels:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        
    def build_custom_cnn_model(self):
        """Build custom CNN model that works offline"""
        print("🏗️ Building Custom CNN Model for Research...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['custom_cnn'] = model
        print("✅ Custom CNN model created successfully")
        return model
    
    def build_simple_model(self):
        """Build a simpler model for quick testing"""
        print("🏗️ Building Simple CNN Model...")
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['simple_cnn'] = model
        print("✅ Simple CNN model created successfully")
        return model
    
    def build_ensemble_model(self):
        """Build ensemble of models for research"""
        print("🏗️ Building Ensemble Model for Research...")
        
        custom_model = self.build_custom_cnn_model()
        simple_model = self.build_simple_model()
        
        ensemble_models = {
            'custom_cnn': custom_model,
            'simple_cnn': simple_model
        }
        
        print("✅ Ensemble model structure created")
        return ensemble_models
    
    def get_model_summary(self):
        """Get comprehensive model summary for research paper"""
        summary = {}
        
        for name, model in self.models.items():
            summary[name] = {
                'total_params': model.count_params(),
                'trainable_params': sum([w.shape.num_elements() for w in model.trainable_weights]),
                'layers': len(model.layers)
            }
        
        return summary

research_models = ResearchMedicalModels()
ensemble_models = research_models.build_ensemble_model()

# Get model summary for research paper
model_summary = research_models.get_model_summary()
print("\n📊 Research Models Summary:")
for model_name, info in model_summary.items():
    print(f"   {model_name.upper()}:")
    print(f"      - Total Parameters: {info['total_params']:,}")
    print(f"      - Trainable Parameters: {info['trainable_params']:,}")
    print(f"      - Layers: {info['layers']}")

# Step 3: Research Evaluation
print("\n📈 Step 3: Comprehensive Research Evaluation...")

class ResearchEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names
        
    def comprehensive_evaluation(self, y_true, y_pred, y_pred_proba):
        """Comprehensive evaluation for research paper"""
        print("📊 Calculating research metrics...")
        
        # Classification report
        clf_report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC for multi-class
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate average AUC
        roc_auc["macro"] = np.mean(list(roc_auc.values()))
        
        results = {
            'classification_report': clf_report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        print("✅ Research metrics calculated")
        return results
    
    def generate_research_metrics(self, results):
        """Generate research-grade metrics table"""
        report = results['classification_report']
        roc_auc = results['roc_auc']
        
        metrics_data = []
        for i, cls in enumerate(self.class_names):
            metrics_data.append({
                'Class': cls,
                'Precision': f"{report[cls]['precision']:.3f}",
                'Recall': f"{report[cls]['recall']:.3f}",
                'F1-Score': f"{report[cls]['f1-score']:.3f}",
                'ROC-AUC': f"{roc_auc[i]:.3f}",
                'Support': report[cls]['support']
            })
        
        # Add macro average
        metrics_data.append({
            'Class': 'Macro Avg',
            'Precision': f"{report['macro avg']['precision']:.3f}",
            'Recall': f"{report['macro avg']['recall']:.3f}",
            'F1-Score': f"{report['macro avg']['f1-score']:.3f}",
            'ROC-AUC': f"{roc_auc['macro']:.3f}",
            'Support': report['macro avg']['support']
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        return metrics_df
    
    def plot_research_figures(self, results, save_path='research_paper/figures'):
        """Generate research-quality figures"""
        os.makedirs(save_path, exist_ok=True)
        print("🖼️ Generating research figures...")
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Medical AI Research', fontsize=16, fontweight='bold')
        plt.ylabel('True Diagnosis', fontsize=12)
        plt.xlabel('AI Prediction', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: ROC Curves
        plt.figure(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Professional colors
        
        for i in range(len(self.class_names)):
            plt.plot(results['fpr'][i], results['tpr'][i],
                    color=colors[i], linewidth=2.5,
                    label=f'{self.class_names[i]} (AUC = {results["roc_auc"][i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Medical Diagnosis', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Research figures saved to {save_path}")

# Simulate research data (1000 samples)
print("🎲 Generating simulated research data...")
np.random.seed(42)  # For reproducible results
y_true = np.random.randint(0, 3, 1000)
y_pred = np.random.randint(0, 3, 1000)
y_pred_proba = np.random.dirichlet(np.ones(3), 1000)

print("📈 Performing comprehensive evaluation...")
evaluator = ResearchEvaluator(class_names)
research_results = evaluator.comprehensive_evaluation(y_true, y_pred, y_pred_proba)

# Generate research metrics table
print("📋 Generating research metrics table...")
metrics_df = evaluator.generate_research_metrics(research_results)

# Generate research figures
evaluator.plot_research_figures(research_results)

# Final Research Summary
print("\n" + "🎯 MEDICAL AI RESEARCH PROJECT SUMMARY" + "🎯")
print("=" * 55)
print(f"📁 Dataset: COVID-19 Chest X-ray Research Collection")
print(f"🔬 Classes: {class_names}")
print(f"🤖 Models: {len(ensemble_models)} ensemble models")
print(f"📊 Evaluation: 1,000 research samples")
print(f"⭐ Macro AUC: {research_results['roc_auc']['macro']:.3f}")
print(f"📈 Figures: Generated in research_paper/figures/")
print(f"📋 Metrics Table:")

# Display metrics table
print("\n" + "="*80)
print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Support':<10}")
print("-"*80)
for _, row in metrics_df.iterrows():
    print(f"{row['Class']:<12} {row['Precision']:<10} {row['Recall']:<10} {row['F1-Score']:<10} {row['ROC-AUC']:<10} {row['Support']:<10}")

print("="*80)
print("\n✅ Research project completed successfully!")
print("🚀 Next: Add real medical data and conduct actual training")
print("💡 This demonstrates the complete research framework for Dr. Kybic")
