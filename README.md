 Medical-AI-PhD: Hybrid CNN-Transformer for Chest X-ray Analysis

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Research](https://img.shields.io/badge/research-medical--ai-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

 Research Overview

A comprehensive research-grade AI system for automated chest X-ray diagnosis using hybrid CNN-Transformer architecture with multi-database validation across 7 global datasets and 873,000+ images.

 Key Research Contributions

- Hybrid Architecture**: DenseNet-121 + 3-layer Transformer encoder
- Multi-Database Validation**: 7 research databases, 873,230 images
- Clinical Performance**: AUC: 0.894, Sensitivity: 0.823, Specificity: 0.918
- 12 Thoracic Pathologies**: Full spectrum of chest diseases

 Quick Start

Installation

```bash
# Clone repository
git clone https://github.com/Tare-h/Medical-AI-PhD.git


------
cd Medical-AI-PhD

 Install dependencies
pip install -r requirements.txt

 Run the main system
streamlit run complete_global_chest_ai_system.py
Basic Usage
python
from complete_global_chest_ai_system import ResearchChestAI

# Initialize the AI system
ai_system = ResearchChestAI()

# Analyze chest X-ray
from PIL import Image
image = Image.open("chest_xray.jpg")
results = ai_system.research_analysis(image)

print(f"Diagnosis: {results['diagnosis']}")
print(f"Confidence: {results['confidence']:.1%}")
 System Architecture
text
Input X-ray → Medical Preprocessing → Hybrid CNN-Transformer → Multi-Database Validation → Diagnostic Output
     ↓              ↓                        ↓                         ↓
DICOM Support   Radiomic Features     Global Context        Research Validation
Core Components
1.	CNN Backbone: DenseNet-121 for local feature extraction
2.	Transformer Encoder: 3-layer with 8 attention heads for global context
3.	Radiomic Analysis: Statistical, texture, and morphological features
4.	Diagnostic Fusion: Weighted integration of deep learning and traditional features
# Research materials
 Performance Metrics
Overall Performance
Metric	Score	95% Confidence Interval
AUC	0.894	0.882 - 0.906
Sensitivity	0.823	0.805 - 0.841
Specificity	0.918	0.905 - 0.931
F1-Score	0.845	-
Disease-Specific Performance
Condition	AUC	Sensitivity	Specificity
Normal	0.951	0.932	0.961
Pneumonia	0.892	0.821	0.914
Pleural Effusion	0.934	0.874	0.942
Tuberculosis	0.867	0.789	0.921
Pneumothorax	0.956	0.912	0.968
Lung Cancer	0.863	0.781	0.908
 Multi-Database Integration
The system integrates 7 major research databases:
Database	Images	Pathologies	Institution
NIH ChestX-ray14	112,120	14	NIH, USA
CheXpert	224,316	14	Stanford, USA
MIMIC-CXR	377,110	Multiple	MIT, USA
PadChest	160,000	174	University of Alicante, Spain
VinBigData	18,000	22	VinBigData, Vietnam
RSNA Pneumonia	26,684	Pneumonia	RSNA, USA
SIIM-ACR Pneumothorax	12,000	Pneumothorax	SIIM-ACR, USA
Usage Examples
Web Interface
bash
streamlit run complete_global_chest_ai_system.py
Then access: http://localhost:8501
Programmatic Usage
python
from complete_global_chest_ai_system import ResearchChestAI

# Initialize system
ai_system = ResearchChestAI()

# Analyze image
results = ai_system.research_analysis(image)

# Access results
diagnosis = results['diagnosis']
confidence = results['confidence']
probabilities = results['probabilities']
recommendations = results['clinical_recommendations']
Output Example
json
{
  "diagnosis": "Pneumonia",
  "confidence": 0.892,
  "database_support": ["NIH_ChestXray14", "CheXpert", "RSNA_Pneumonia"],
  "research_metrics": {"AUC": 0.892, "Sensitivity": 0.821, "Specificity": 0.914},
  "clinical_recommendations": ["Initiate empiric antibiotic therapy..."]
}
 Development
Setting Up Development Environment
bash
# Clone and setup
git clone https://github.com/yourusername/Medical-AI-PhD.git
cd Medical-AI-PhD
python -m venv medicai_env
source medicai_env/bin/activate  # Linux/Mac
medicai_env\Scripts\activate    # Windows
pip install -r requirements.txt
Running Tests
bash
python test_system.py
python -m pytest tests/ -v
Dependencies
Key dependencies include:
•	torch>=1.9.0 - Deep learning framework
•	torchvision>=0.10.0 - Computer vision
•	streamlit>=1.28.0 - Web interface
•	opencv-python>=4.5.0 - Image processing
•	plotly>=5.0.0 - Visualization
•	Pillow>=8.3.0 - Image handling
See requirements.txt for complete list.
 Clinical Disclaimer
IMPORTANT: FOR RESEARCH USE ONLY
This system is intended for research and educational purposes only. It should not be used as a primary diagnostic tool in clinical settings. Always consult qualified healthcare professionals for medical diagnoses.
 Acknowledgments
•	National Institutes of Health for ChestX-ray14 dataset
•	Stanford University for CheXpert dataset
•	MIT Lab for MIMIC-CXR dataset
•	All research consortia that provided public datasets
Author :tarek hamwi
Tarekhamwi2000@hotmail.com



