import cv2
import numpy as np
from PIL import Image, ImageEnhance

class MedicalImageProcessor:
    def __init__(self):
        self.standard_size = (224, 224)
    
    def enhance_image_quality(self, image):
        image = image.convert('RGB')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def validate_medical_image(self, image):
        if image.mode not in ['RGB', 'L']:
            raise ValueError("Image format not supported")
        
        if image.size[0] < 100 or image.size[1] < 100:
            raise ValueError("Image resolution too low")
        
        return True
    
    def prepare_for_analysis(self, image):
        try:
            self.validate_medical_image(image)
            enhanced_image = self.enhance_image_quality(image)
            return enhanced_image
        except Exception as e:
            raise Exception(f"Image processing error: {str(e)}")
