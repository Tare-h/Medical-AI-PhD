import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

class ResearchMedicalModels:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        
    def build_densenet_model(self):
        """Build DenseNet-121 based model (State-of-the-art for medical imaging)"""
        base_model = applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers, fine-tune later layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
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
        
        self.models['densenet'] = model
        return model
    
    def build_resnet_model(self):
        """Build ResNet-50 based model"""
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['resnet'] = model
        return model
    
    def build_ensemble_model(self):
        """Build ensemble of multiple models for research"""
        print("🏗️ Building Ensemble Model for Research...")
        
        densenet_model = self.build_densenet_model()
        resnet_model = self.build_resnet_model()
        
        ensemble_models = {
            'densenet': densenet_model,
            'resnet': resnet_model
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
                'non_trainable_params': sum([w.shape.num_elements() for w in model.non_trainable_weights]),
                'layers': len(model.layers)
            }
        
        return summary

# Test research models
if __name__ == "__main__":
    research_models = ResearchMedicalModels()
    
    # Build ensemble model
    ensemble = research_models.build_ensemble_model()
    
    # Get model summary for research paper
    summary = research_models.get_model_summary()
    
    print("🔬 Research Models Summary:")
    for model_name, info in summary.items():
        print(f"   {model_name.upper()}:")
        print(f"      - Total Parameters: {info['total_params']:,}")
        print(f"      - Trainable Parameters: {info['trainable_params']:,}")
        print(f"      - Layers: {info['layers']}")
