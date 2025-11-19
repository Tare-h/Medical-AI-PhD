"""
Create and save scaler for medical app
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processing import MedicalDataProcessor
import os

def create_and_save_scaler():
    """Create and save fitted scaler"""
    print("🔄 Creating and fitting scaler...")
    
    try:
        # Load and preprocess data
        processor = MedicalDataProcessor()
        data = processor.load_and_validate_data()
        X, y = processor.preprocess_data(data)
        
        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save scaler
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(scaler, 'saved_models/scaler.pkl')
        
        print("✅ Scaler created and saved successfully!")
        print(f"   Data shape: {X.shape}")
        print(f"   Scaler saved to: saved_models/scaler.pkl")
        
    except Exception as e:
        print(f"❌ Error creating scaler: {e}")

if __name__ == "__main__":
    create_and_save_scaler()
