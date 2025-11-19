import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class ResearchTrainingFramework:
    def __init__(self):
        self.training_logs = {}
        self.research_results = {}
        self.start_time = datetime.now()
    
    def setup_research_training(self, model, model_name):
        """
        Setup research-grade training configuration
        Implements best practices from medical AI research
        """
        print(f"Setting up research training for {model_name}...")
        
        # Research callbacks for optimal training
        callbacks = [
            # Model checkpointing
            ModelCheckpoint(
                f'research_paper/model_checkpoints/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping with research patience
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduling
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def conduct_research_experiment(self, model, model_name, training_data, validation_data, epochs=100):
        """
        Conduct formal research experiment with comprehensive logging
        """
        print(f"Starting research experiment: {model_name}")
        
        # Setup training configuration
        callbacks = self.setup_research_training(model, model_name)
        
        # Research training execution
        print(f"Training {model_name} for {epochs} epochs...")
        
        history = model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log research results
        self.training_logs[model_name] = history.history
        self.save_research_training_logs(model_name, history.history)
        
        print(f"Research experiment completed: {model_name}")
        return history
    
    def save_research_training_logs(self, model_name, history):
        """
        Save comprehensive training logs for research documentation
        """
        log_data = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'training_duration': str(datetime.now() - self.start_time),
            'final_training_accuracy': history['accuracy'][-1],
            'final_validation_accuracy': history['val_accuracy'][-1],
            'final_training_loss': history['loss'][-1],
            'final_validation_loss': history['val_loss'][-1],
            'learning_curve': {
                'epochs': list(range(len(history['accuracy']))),
                'training_accuracy': history['accuracy'],
                'validation_accuracy': history['val_accuracy'],
                'training_loss': history['loss'],
                'validation_loss': history['val_loss']
            }
        }
        
        # Save to research logs
        with open(f'research_paper/training_logs/{model_name}_training_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def generate_research_training_report(self):
        """
        Generate comprehensive research training report
        """
        print("Generating research training report...")
        
        report = {
            'research_project': 'MedicAI Chest X-Ray Diagnosis System',
            'training_completion_date': datetime.now().isoformat(),
            'total_training_time': str(datetime.now() - self.start_time),
            'models_trained': list(self.training_logs.keys()),
            'performance_summary': self.calculate_performance_summary(),
            'training_analysis': self.analyze_training_performance()
        }
        
        # Save research report
        with open('research_paper/research_training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Research training report generated successfully")
        return report
    
    def calculate_performance_summary(self):
        """Calculate comprehensive performance metrics"""
        performance = {}
        
        for model_name, history in self.training_logs.items():
            performance[model_name] = {
                'best_training_accuracy': max(history['accuracy']),
                'best_validation_accuracy': max(history['val_accuracy']),
                'final_training_accuracy': history['accuracy'][-1],
                'final_validation_accuracy': history['val_accuracy'][-1],
                'training_loss': history['loss'][-1],
                'validation_loss': history['val_loss'][-1]
            }
        
        return performance
    
    def analyze_training_performance(self):
        """Analyze training performance for research insights"""
        analysis = {
            'overfitting_analysis': {},
            'convergence_analysis': {},
            'model_comparison': {}
        }
        
        for model_name, history in self.training_logs.items():
            # Overfitting analysis
            final_train_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            overfitting_gap = final_train_acc - final_val_acc
            
            analysis['overfitting_analysis'][model_name] = {
                'training_accuracy': final_train_acc,
                'validation_accuracy': final_val_acc,
                'accuracy_gap': overfitting_gap,
                'overfitting_level': 'High' if overfitting_gap > 0.1 else 'Moderate' if overfitting_gap > 0.05 else 'Low'
            }
        
        return analysis

if __name__ == "__main__":
    research_trainer = ResearchTrainingFramework()
    print("Research training framework initialized successfully")
