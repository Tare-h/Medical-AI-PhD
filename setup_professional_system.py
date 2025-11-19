# setup_professional_system.py
import os
import shutil

def create_professional_structure():
    """Create professional research structure for European conferences"""
    
    print("🏗️ Creating Professional Research Structure...")
    
    # Remove old files
    old_files = [
        "advanced_report_generator.py.py",
        "data_collaction.py",
        "data_pilitter.py", 
        "dIcon_processor.py",
        "enhanced biomarkers.py",
        "phase model building.py",
        "phase_simple_log"
    ]
    
    for file in old_files:
        if os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
            else:
                shutil.rmtree(file)
            print(f"🗑️ Deleted: {file}")
    
    # Professional research structure
    professional_structure = {
        "ChestXAI_Research": {
            "__init__.py": "# Chest X-ray AI Research System",
            "config": {
                "__init__.py": "",
                "research_config.py": "# Research configuration"
            },
            "models": {
                "__init__.py": "",
                "hybrid_transformer_cnn.py": "# Main hybrid architecture"
            },
            "data": {
                "__init__.py": "",
                "preprocessing.py": "# Data preprocessing"
            },
            "training": {
                "__init__.py": "",
                "trainer.py": "# Training pipeline"
            },
            "evaluation": {
                "__init__.py": "",
                "metrics.py": "# Comprehensive metrics"
            },
            "utils": {
                "__init__.py": "",
                "logger.py": "# Experiment logging"
            }
        },
        "api": {
            "__init__.py": "",
            "research_dashboard.py": "# Main research dashboard"
        },
        "requirements.txt": "# Python dependencies",
        "run_demo.py": "# Demo launcher"
    }
    
    def create_structure(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                create_structure(path, content)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"📁 Created: {path}")
    
    create_structure(".", professional_structure)
    print("✅ Professional structure created successfully!")

if __name__ == "__main__":
    create_professional_structure()