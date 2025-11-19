"""
Medical AI REST API
FastAPI backend for medical diagnosis
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict
import uvicorn

# Pydantic models for request/response
class PatientData(BaseModel):
    age: float
    blood_pressure: float
    cholesterol: float
    glucose: float
    bmi: float
    heart_rate: float

class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    probabilities: Dict[str, float]
    risk_level: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI Diagnosis API",
    description="REST API for medical diagnosis prediction",
    version="1.0.0"
)

# Global variables
model = None
scaler = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global model, scaler
    try:
        model = joblib.load('saved_models/best_optimized_simple.pkl')
        # In production, you'd load a saved scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_diagnosis(patient_data: PatientData):
    """Predict medical diagnosis"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([
            patient_data.age,
            patient_data.blood_pressure,
            patient_data.cholesterol,
            patient_data.glucose,
            patient_data.bmi,
            patient_data.heart_rate
        ]).reshape(1, -1)
        
        # Scale features (in production, use fitted scaler)
        input_scaled = scaler.fit_transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Map prediction to diagnosis
        diagnoses = ['Disease_A', 'Disease_B', 'Healthy']
        diagnosis = diagnoses[prediction]
        confidence = probabilities[prediction]
        
        # Determine risk level
        if diagnosis == 'Healthy':
            risk_level = 'low'
        elif confidence > 0.7:
            risk_level = 'high'
        else:
            risk_level = 'medium'
        
        # Create response
        prob_dict = {diagnoses[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return PredictionResponse(
            diagnosis=diagnosis,
            confidence=float(confidence),
            probabilities=prob_dict,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "features": ['age', 'blood_pressure', 'cholesterol', 'glucose', 'bmi', 'heart_rate'],
        "classes": ['Disease_A', 'Disease_B', 'Healthy']
    }
    
    if hasattr(model, 'feature_importances_'):
        info["feature_importance"] = dict(zip(
            info["features"], 
            model.feature_importances_.tolist()
        ))
    
    return info

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
