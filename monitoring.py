"""
Model Monitoring and Performance Tracking
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import json

class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self):
        self.performance_log = []
        self.drift_detector = None
    
    def log_prediction(self, features, prediction, confidence, actual=None):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'confidence': confidence,
            'actual': actual
        }
        
        self.performance_log.append(log_entry)
        
        # Keep only last 10,000 entries
        if len(self.performance_log) > 10000:
            self.performance_log = self.performance_log[-10000:]
    
    def calculate_performance_metrics(self):
        """Calculate current performance metrics"""
        if not self.performance_log:
            return {}
        
        df = pd.DataFrame(self.performance_log)
        
        metrics = {
            'total_predictions': len(df),
            'average_confidence': df['confidence'].mean(),
            'low_confidence_predictions': len(df[df['confidence'] < 0.7]),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def save_monitoring_data(self):
        """Save monitoring data to file"""
        with open('monitoring/performance_log.json', 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        
        metrics = self.calculate_performance_metrics()
        with open('monitoring/current_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
