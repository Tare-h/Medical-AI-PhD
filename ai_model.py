import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, models

class MedicalAIModel:
    def __init__(self):
        self.model = None
        self.input_size = (224, 224)
        self.class_names = {
            'covid': 'COVID-19',
            'pneumonia': 'Pneumonia', 
            'normal': 'Normal'
        }
        
    def create_advanced_model(self):
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        image = image.convert('RGB')
        image = image.resize(self.input_size)
        img_array = np.array(image)
        
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_disease(self, image):
        processed_image = self.preprocess_image(image)
        
        if self.model is None:
            self.create_advanced_model()
        
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        class_keys = list(self.class_names.keys())
        predicted_class = class_keys[predicted_class_idx]
        
        return self.class_names[predicted_class], confidence
