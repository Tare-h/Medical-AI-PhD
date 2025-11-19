"""
Professional Medical Data Processing Module
Phase 4 - Medical AI PhD Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, Optional, List

from config import DATA_CONFIG

class MedicalDataProcessor:
    """
    Professional medical data processor for handling clinical datasets
    with comprehensive preprocessing and validation
    """
    
    def __init__(self, config=DATA_CONFIG):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.logger = logging.getLogger("MedicalAI.DataProcessor")
        self.is_fitted = False
        
        self.logger.info("MedicalDataProcessor initialized successfully")
    
    def load_and_validate_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load medical data with comprehensive validation
        """
        data_path = file_path or self.config.DATA_PATH
        
        try:
            self.logger.info(f"Loading data from: {data_path}")
            data = pd.read_csv(data_path)
            
            # Basic validation
            self._validate_data_structure(data)
            
            self.logger.info(f"Data loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")
            self.logger.info(f"Feature columns: {self.config.FEATURE_COLUMNS}")
            self.logger.info(f"Target column: {self.config.TARGET_COLUMN}")
            
            return data
            
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data_structure(self, data: pd.DataFrame):
        """Validate data structure and required columns"""
        required_columns = self.config.FEATURE_COLUMNS + [self.config.TARGET_COLUMN]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for sufficient samples
        if len(data) < 10:
            raise ValueError("Insufficient data samples")
        
        self.logger.info("Data structure validation passed")
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Comprehensive data preprocessing pipeline
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        # Extract features and target
        X = data[self.config.FEATURE_COLUMNS].copy()
        y = data[self.config.TARGET_COLUMN].copy()
        
        # Handle missing values
        X_imputed = self._handle_missing_values(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.is_fitted = True
        
        self.logger.info(f"Preprocessing completed: X.shape={X_scaled.shape}, y.shape={y_encoded.shape}")
        self.logger.info(f"Target classes: {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded
    
    def _handle_missing_values(self, X: pd.DataFrame) -> np.ndarray:
        """Handle missing values in features"""
        missing_count = X.isnull().sum().sum()
        
        if missing_count > 0:
            self.logger.warning(f"Found {missing_count} missing values, applying imputation")
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = X.values
        
        return X_imputed
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train, validation, and test sets
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Second split: separate validation set from training
        val_size = self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp
        )
        
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Training:   {X_train.shape[0]} samples")
        self.logger.info(f"  Validation: {X_val.shape[0]} samples")
        self.logger.info(f"  Test:       {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_names(self) -> List[str]:
        """Get feature column names"""
        return self.config.FEATURE_COLUMNS
    
    def get_target_names(self) -> List[str]:
        """Get target class names"""
        if self.is_fitted:
            return list(self.label_encoder.classes_)
        else:
            return [self.config.TARGET_COLUMN]
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """Generate comprehensive data summary"""
        summary = {
            'total_samples': len(data),
            'total_features': len(data.columns),
            'feature_columns': self.config.FEATURE_COLUMNS,
            'target_column': self.config.TARGET_COLUMN,
            'target_distribution': data[self.config.TARGET_COLUMN].value_counts().to_dict(),
            'missing_values': data[self.config.FEATURE_COLUMNS].isnull().sum().to_dict(),
            'data_types': data[self.config.FEATURE_COLUMNS].dtypes.astype(str).to_dict()
        }
        return summary
