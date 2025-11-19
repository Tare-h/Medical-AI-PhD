"""
Advanced Model Factory for Medical AI PhD
Phase 5: Multi-Model Development and Comparison
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Any, Tuple
import joblib
import logging
import os

class MedicalModelFactory:
    """
    Advanced factory for creating, training, and comparing multiple medical AI models
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MedicalAI.ModelFactory")
        self.models = {}
        self.results = {}
        self.evaluation_results = {}
        
    def create_sklearn_models(self) -> Dict:
        """Create comprehensive set of traditional ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'svm_rbf': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                multi_class='multinomial'
            ),
            'mlp_classifier': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        self.logger.info("✅ Created 5 advanced sklearn models for comprehensive comparison")
        return self.models
    
    def create_deep_learning_model(self, input_dim: int, num_classes: int) -> keras.Model:
        """Create advanced deep learning model with modern architecture"""
        
        model = keras.Sequential([
            # Input layer with batch normalization
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            # Hidden layers with progressive complexity reduction
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            
            # Output layer
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Advanced optimizer configuration
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.models['deep_learning'] = model
        self.logger.info(f"✅ Created advanced deep learning model: {input_dim} -> {num_classes}")
        return model
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Comprehensive training of all models with validation"""
        self.results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"🔄 Training {name}...")
            
            try:
                if name == 'deep_learning':
                    # Advanced training for deep learning
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=150,
                        batch_size=32,
                        verbose=0,
                        callbacks=[
                            keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=20,
                                restore_best_weights=True,
                                verbose=1
                            ),
                            keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,
                                patience=10,
                                min_lr=1e-7,
                                verbose=1
                            )
                        ]
                    )
                    self.results[name] = {
                        'model': model,
                        'history': history.history,
                        'type': 'deep_learning'
                    }
                else:
                    # Traditional ML models
                    model.fit(X_train, y_train)
                    self.results[name] = {
                        'model': model,
                        'history': None,
                        'type': 'sklearn'
                    }
                    
                self.logger.info(f"✅ {name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train {name}: {str(e)}")
                continue
        
        return self.results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive evaluation of all trained models"""
        self.evaluation_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            
            try:
                if result['type'] == 'deep_learning':
                    # Deep learning evaluation
                    y_pred_proba = model.predict(X_test, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    y_scores = y_pred_proba
                else:
                    # Sklearn evaluation
                    y_pred = model.predict(X_test)
                    y_scores = model.predict_proba(X_test)
                
                # Calculate comprehensive metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_scores, multi_class='ovr', average='weighted')
                }
                
                self.evaluation_results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_scores
                }
                
                self.logger.info(
                    f"📊 {name} - Accuracy: {metrics['accuracy']:.4f}, "
                    f"F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"❌ Evaluation failed for {name}: {str(e)}")
                continue
        
        return self.evaluation_results
    
    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Any, Dict]:
        """Get the best performing model based on specified metric"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_models first.")
        
        best_model_name = max(
            self.evaluation_results.keys(),
            key=lambda x: self.evaluation_results[x]['metrics'].get(metric, 0)
        )
        
        best_model = self.results[best_model_name]['model']
        best_metrics = self.evaluation_results[best_model_name]['metrics']
        
        self.logger.info(
            f"🏆 Best model: {best_model_name} "
            f"({metric}: {best_metrics[metric]:.4f})"
        )
        
        return best_model_name, best_model, best_metrics
    
    def save_models(self, directory: str = 'saved_models'):
        """Save all trained models with metadata"""
        os.makedirs(directory, exist_ok=True)
        
        for name, result in self.results.items():
            model = result['model']
            try:
                if result['type'] == 'deep_learning':
                    # Save Keras model
                    model_path = os.path.join(directory, f'{name}_model.h5')
                    model.save(model_path)
                else:
                    # Save sklearn model
                    model_path = os.path.join(directory, f'{name}_model.pkl')
                    joblib.dump(model, model_path)
                
                self.logger.info(f"💾 Saved {name} to {model_path}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to save {name}: {str(e)}")
        
        # Save overall comparison
        comparison_path = os.path.join(directory, 'model_comparison.csv')
        comparison_df = self.get_model_comparison_dataframe()
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"📈 Model comparison saved to {comparison_path}")
    
    def get_model_comparison_dataframe(self) -> pd.DataFrame:
        """Create comprehensive comparison DataFrame"""
        comparison_data = []
        
        for name, eval_result in self.evaluation_results.items():
            metrics = eval_result['metrics']
            row = {'Model': name}
            row.update(metrics)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
