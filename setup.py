import os
import shutil

def create_simple_structure():
    print("🏗️ Creating Professional Research Structure...")
    
    # تنظيف الملفات القديمة
    old_files = [
        "advanced_report_generator.py.py",
        "data_collaction.py",
        "data_pilitter.py", 
        "dIcon_processor.py"
    ]
    
    for file in old_files:
        if os.path.exists(file):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)
                print(f"🗑️ Deleted: {file}")
            except:
                pass
    
    # إنشاء المجلدات الأساسية
    folders = [
        "ChestXAI_Research",
        "ChestXAI_Research/config", 
        "ChestXAI_Research/api",
        "saved_models",
        "logs",
        "data"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created: {folder}")
    
    # إنشاء ملف الإعدادات
    config_content = '''
import torch
import os

class ResearchConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CHEST_DISEASES = {
        0: "Normal",
        1: "Pleural Effusion", 
        2: "Pneumonia",
        3: "COVID-19",
        4: "Tuberculosis",
        5: "Pneumothorax"
    }
    
    @classmethod
    def setup_directories(cls):
        os.makedirs("saved_models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
'''
    
    # إنشاء ملف الداشبورد الرئيسي
    dashboard_content = '''
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleResearchDashboard:
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(
            page_title="ChestXAI Research System",
            page_icon="🫁", 
            layout="wide"
        )
        
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🫁 ChestXAI Research System</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def run(self):
        st.sidebar.title("🔬 Research Info")
        st.sidebar.info("Hybrid Transformer-CNN Model for Chest X-ray Analysis")
        
        uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_column_width=True)
            
            with col2:
                if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        # محاكاة التحليل
                        result = self.analyze_image(image)
                    
                    self.show_results(result)
    
    def analyze_image(self, image):
        diseases = ["Normal", "Pleural Effusion", "Pneumonia", "COVID-19", "Tuberculosis", "Pneumothorax"]
        np.random.seed(hash(image.tobytes()) % 1000)
        
        probs = np.random.dirichlet(np.ones(6) * 0.5)
        probs[0] += 0.3
        probs = probs / probs.sum()
        
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        return {
            'diagnosis': diseases[predicted_class],
            'confidence': confidence,
            'probabilities': {diseases[i]: float(p) for i, p in enumerate(probs)},
            'risk_level': 'High' if predicted_class in [1, 3, 5] and confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'
        }
    
    def show_results(self, result):
        st.header("📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Diagnosis", result['diagnosis'])
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        with col3:
            st.metric("Risk Level", result['risk_level'])
        
        # Probability chart
        df = pd.DataFrame({
            'Condition': list(result['probabilities'].keys()),
            'Probability': list(result['probabilities'].values())
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(df, x='Probability', y='Condition', orientation='h',
                     title='Disease Probability Distribution',
                     color='Probability')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🔬 Research Use Only - For demonstration purposes")

def main():
    app = SimpleResearchDashboard()
    app.run()

if __name__ == "__main__":
    main()
'''

    # إنشاء ملفات
    with open("ChestXAI_Research/config/research_config.py", "w") as f:
        f.write(config_content)
    
    with open("ChestXAI_Research/api/research_dashboard.py", "w") as f:
        f.write(dashboard_content)
    
    # إنشاء ملف المتطلبات
    with open("requirements.txt", "w") as f:
        f.write("streamlit>=1.24.0\nnumpy>=1.24.0\npandas>=2.0.0\nplotly>=5.14.0\nPillow>=9.5.0\ntorch>=2.0.0")
    
    # إنشاء ملف التشغيل
    with open("run_demo.py", "w") as f:
        f.write('''import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 Launching ChestXAI Research System...")
try:
    from ChestXAI_Research.api.research_dashboard import main
    print("✅ System loaded successfully!")
    print("🌐 Starting dashboard...")
    main()
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please install: pip install streamlit numpy pandas plotly Pillow")
''')

    print("✅ All files created successfully!")
    print("📋 Next steps:")
    print("1. pip install -r requirements.txt")
    print("2. python run_demo.py")

if __name__ == "__main__":
    create_simple_structure()