# logger.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
from pathlib import Path

class MedicalLogger:
    """
    Comprehensive logging system for medical AI application
    """
    
    def __init__(self, name="medical_ai", log_level="INFO", config=None):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.config = config or {}
        
        # Create logs directory
        self.logs_dir = Path(self.config.get('logs_dir', 'logs'))
        self.logs_dir.mkdir(exist_ok=True)
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with multiple handlers"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File Handler (Rotating)
        log_file = self.logs_dir / f"medical_ai_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Medical Events Handler (JSON format)
        events_file = self.logs_dir / f"medical_events_{datetime.now().strftime('%Y%m%d')}.json"
        events_handler = RotatingFileHandler(
            events_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        events_handler.setLevel(logging.INFO)
        events_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(events_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def log_medical_event(self, event_type: str, data: dict, level: str = "INFO"):
        """
        Log medical-specific events in JSON format
        """
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'level': level.upper(),
            'data': data
        }
        
        # Find the JSON handler
        json_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler) and 'medical_events' in handler.baseFilename:
                json_handler = handler
                break
        
        if json_handler:
            # Temporarily remove other handlers
            original_handlers = self.logger.handlers.copy()
            self.logger.handlers = [json_handler]
            
            # Log as JSON
            self.logger.info(json.dumps(event_data, ensure_ascii=False))
            
            # Restore handlers
            self.logger.handlers = original_handlers
    
    def log_analysis_start(self, study_id: str, patient_id: str, modality: str):
        """Log analysis start event"""
        self.log_medical_event(
            'analysis_started',
            {
                'study_id': study_id,
                'patient_id': patient_id,
                'modality': modality,
                'timestamp': datetime.now().isoformat()
            },
            'INFO'
        )
        self.logger.info(f"Analysis started - Study: {study_id}, Patient: {patient_id}, Modality: {modality}")
    
    def log_analysis_complete(self, study_id: str, diagnosis: str, confidence: float, processing_time: float):
        """Log analysis completion event"""
        self.log_medical_event(
            'analysis_completed',
            {
                'study_id': study_id,
                'diagnosis': diagnosis,
                'confidence': confidence,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            },
            'INFO'
        )
        self.logger.info(f"Analysis completed - Study: {study_id}, Diagnosis: {diagnosis}, Confidence: {confidence:.2%}, Time: {processing_time:.2f}s")
    
    def log_security_event(self, event: str, user: str = None, details: dict = None):
        """Log security-related events"""
        security_data = {
            'event': event,
            'user': user,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_medical_event('security_event', security_data, 'WARNING')
        self.logger.warning(f"Security event - {event}, User: {user}")
    
    def log_error(self, error_type: str, message: str, details: dict = None):
        """Log error events"""
        error_data = {
            'error_type': error_type,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_medical_event('error', error_data, 'ERROR')
        self.logger.error(f"{error_type} - {message}")
    
    def log_performance(self, operation: str, duration: float, metrics: dict = None):
        """Log performance metrics"""
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_medical_event('performance', perf_data, 'DEBUG')
        self.logger.debug(f"Performance - {operation}: {duration:.3f}s")
    
    def get_log_file_path(self, log_type='main'):
        """Get path to log files"""
        if log_type == 'main':
            return self.logs_dir / f"medical_ai_{datetime.now().strftime('%Y%m%d')}.log"
        elif log_type == 'events':
            return self.logs_dir / f"medical_events_{datetime.now().strftime('%Y%m%d')}.json"
        return None

# Global logger instance
medical_logger = MedicalLogger()

# Convenience functions
def get_logger(name=None):
    """Get logger instance"""
    if name:
        return MedicalLogger(name).logger
    return medical_logger.logger

def setup_logging(config_path="config.yaml"):
    """Setup logging from config file"""
    import yaml
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging_config = config.get('logging', {})
        return MedicalLogger(
            name=logging_config.get('name', 'medical_ai'),
            log_level=logging_config.get('level', 'INFO'),
            config=logging_config
        )
    except Exception as e:
        print(f"Failed to setup logging from config: {e}")
        return MedicalLogger()

# Example usage
if __name__ == "__main__":
    logger = setup_logging()
    
    # Test different log types
    logger.logger.info("System initialized successfully")
    logger.log_analysis_start("STU_001", "PAT_001", "X-Ray")
    logger.log_analysis_complete("STU_001", "Normal", 0.94, 2.34)
    logger.log_performance("image_processing", 1.23, {"images_processed": 5})
    
    print("Logging test completed. Check logs directory.")
