"""
Advanced PDF Report Generator for Medical AI System
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from datetime import datetime
from typing import Dict

class AdvancedPDFReportGenerator:
    """Advanced PDF report generator for medical AI system"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_professional_styles()
    
    def setup_professional_styles(self):
        """Setup professional medical report styles"""
        
        # Main title style
        self.styles.add(ParagraphStyle(
            name='ProfessionalTitle',
            fontName='Helvetica-Bold',
            fontSize=18,
            textColor=colors.HexColor('#2E5A8D'),
            alignment=1,
            spaceAfter=30
        ))
        
        # Section headers
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#1A365D'),
            spaceAfter=15
        ))
    
    def generate_medical_report(self, patient_data: Dict, analysis_data: Dict, output_path: str):
        """Generate medical report"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Main title
            title = Paragraph("Medical Imaging AI Analysis Report", self.styles['ProfessionalTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Patient information
            patient_info = [
                ["Patient Information", "Value"],
                ["Name", patient_data.get('name', 'Not specified')],
                ["Age", str(patient_data.get('age', 'Not specified'))],
                ["Gender", patient_data.get('gender', 'Not specified')],
                ["Report Date", datetime.now().strftime('%Y-%m-%d %H:%M')]
            ]
            
            patient_table = Table(patient_info, colWidths=[200, 200])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # Analysis results
            story.append(Paragraph("AI Analysis Results", self.styles['SectionHeader']))
            
            biomarkers = analysis_data.get('biomarkers', {})
            analysis_results = [
                ["Biomarker", "Value"],
                ["Mean Intensity", f"{biomarkers.get('mean_intensity', 0):.3f}"],
                ["Contrast (STD)", f"{biomarkers.get('std_intensity', 0):.3f}"],
                ["Entropy", f"{biomarkers.get('entropy', 0):.3f}"],
                ["Homogeneity", f"{biomarkers.get('homogeneity', 0):.3f}"],
                ["Edge Density", f"{biomarkers.get('edge_density', 0):.3f}"],
                ["Region Count", str(biomarkers.get('region_count', 0))]
            ]
            
            results_table = Table(analysis_results, colWidths=[150, 150])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(results_table)
            
            # Build PDF
            doc.build(story)
            print(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None

# Create alias for compatibility
PDFReportGenerator = AdvancedPDFReportGenerator
