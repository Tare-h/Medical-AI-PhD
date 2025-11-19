import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

class ResearchMedicalModels:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.research_models = {}
        self.training_history = {}
    
    def build_advanced_ensemble(self):
        """
        Build ensemble of state-of-the-art models for medical imaging
        Based on latest research in medical AI
        """
        print("Constructing research-grade ensemble model...")
        
        models_dict = {
            'densenet_medical': self.build_medical_densenet(),
            'efficientnet_medical': self.build_medical_efficientnet(),
            'custom_resnet': self.build_custom_resnet()
        }
        
        self.research_models = models_dict
        print("Research ensemble model construction completed")
        return models_dict
    
    def build_medical_densenet(self):
        """DenseNet-121 optimized for medical imaging"""
        base_model = applications.DenseNet121(
            weights=None,  # Train from scratch for medical domain
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Custom top layers for medical classification
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Research-optimized compiler settings
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def build_medical_efficientnet(self):
        """EfficientNet-B3 for medical image analysis"""
        base_model = applications.EfficientNetB3(
            weights=None,
            include_top=False,
            input_shape=self.input_shape
        )
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(896, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(448, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy', 
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def build_custom_resnet(self):
        """Custom ResNet architecture for medical images"""
        def residual_block(x, filters, kernel_size=3):
            # Skip connection
            skip = x
            
            # First convolution
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Second convolution  
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Add skip connection
            if skip.shape[-1] != filters:
                skip = layers.Conv2D(filters, 1, padding='same')(skip)
            
            x = layers.Add()([x, skip])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 128)
        x = residual_block(x, 256)
        x = residual_block(x, 512)
        
        # Classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model

if __name__ == "__main__":
    research_models = ResearchMedicalModels()
    ensemble = research_models.build_advanced_ensemble()
    print("Advanced research models ready for training")
