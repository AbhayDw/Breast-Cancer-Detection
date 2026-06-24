import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ==========================================
# 1. INITIAL SETUP, CORE ARTIFACTS & SAMPLES
# ==========================================
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Realistic sample patient data from the Wisconsin Dataset
SAMPLE_PATIENT = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

try:
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model assets: {e}")
    model, scaler = None, None

# ==========================================
# 2. CORE UTILITY FUNCTIONS & EXPORTS
# ==========================================
def generate_pdf_report(total, benign, malignant, df):
    """Generates a professional diagnostic PDF report using ReportLab."""
    pdf_path = "Diagnostic_Batch_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(f"<b>Breast Cancer Detection Analytics Report</b>", styles['Title']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Sample Records Evaluated", str(total)],
        ["Total Benign Cases Identified", str(benign)],
        ["Total Malignant Cases Identified", str(malignant)]
    ]
    t_summary = Table(summary_data, colWidths=[250, 100])
    t_summary.setStyle(TableStyle([('BACKGROUND', (0,0), (1,0), colors.teal), ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.grey)]))
    story.append(t_summary)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Preview of Batch Diagnostics (Top 10 Records)</b>", styles['Heading3']))
    preview_rows = [["Record Index", "Final Diagnosis", "Confidence Score (%)"]]
    for idx, row in df.head(10).iterrows():
        preview_rows.append([f"Patient #{idx + 1}", str(row['Prediction']), f"{row['Probability (%)']}%"])
        
    t_preview = Table(preview_rows, colWidths=[120, 120, 120])
    t_preview.setStyle(TableStyle([('BACKGROUND', (0,0), (2,0), colors.grey), ('TEXTCOLOR', (0,0), (2,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)]))
    story.append(t_preview)
    
    doc.build(story)
    return pdf_path

# ==========================================
# 3. PREDICTION & ACTION HANDLERS
# ==========================================
def predict_single(*inputs):
    if model is None or scaler is None: 
        return {"Error": 1.0}
    try:
        scaled = scaler.transform(np.array(inputs).reshape(1, -1))
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]
        
        # Gradio Label dictionary format: {"Label_Name": probability_float}
        # Gradio automatically handles the colors and formatting for labels!
        label_name = "Malignant" if pred == 1 else "Benign"
        return {label_name: float(prob[pred])}
    except Exception as e:
        return {f"Pipeline Error: {str(e)}": 1.0}

def predict_batch(file):
    if model is None or scaler is None or file is None: 
        return None, None, "File upload missing or model error.", 0, 0, 0, 0.0
    try:
        df = pd.read_csv(file.name)
        if any(col not in df.columns for col in FEATURES): 
            return None, None, "Error: CSV schema layout missing required features.", 0, 0, 0, 0.0
        
        scaled = scaler.transform(df[FEATURES])
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)
        
        df['Prediction'] = ["Malignant" if p == 1 else "Benign" for p in preds]
        df['Probability (%)'] = [round(max(pr) * 100, 2) for pr in probs]
        
        total = len(df)
        b_count = int(np.sum(preds == 0))
        m_count = int(np.sum(preds == 1))
        avg_conf = round(df['Probability (%)'].mean(), 2)
        
        csv_path = "batch_predictions.csv"
        df.to_csv(csv_path, index=False)
        pdf_path = generate_pdf_report(total, b_count, m_count, df)
        
        return csv_path, pdf_path, "Batch processing complete!", total, b_count, m_count, avg_conf
    except Exception as e:
        return None, None, f"Parsing Failure: {str(e)}", 0, 0, 0, 0.0

# ==========================================
# 4. GRADIO INTERFACE LAYOUT (BLOCKS)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# 🔬 Breast Cancer Detection Dashboard")
    
    with gr.Tabs():
        # TAB 1: Single Diagnostic Metrics UI
        with gr.TabItem("Single Prediction"):
            inputs_list = []
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("📊 Mean Features", open=True):
                        for f in FEATURES[0:10]: 
                            inputs_list.append(gr.Number(label=f.replace('_', ' ').title(), value=0.0))
                with gr.Column():
                    with gr.Accordion("📉 Standard Error Features", open=False):
                        for f in FEATURES[10:20]: 
                            inputs_list.append(gr.Number(label=f.replace('_', ' ').title(), value=0.0))
                with gr.Column():
                    with gr.Accordion("🔺 Worst Features", open=False):
                        for f in FEATURES[20:30]: 
                            inputs_list.append(gr.Number(label=f.replace('_', ' ').title(), value=0.0))
            
            with gr.Row():
                sample_btn = gr.Button("📋 Load Sample Patient", variant="secondary")
                clear_btn = gr.Button("🔄 Clear All", variant="stop")
                predict_btn = gr.Button("Analyze Metrics", variant="primary")
                
            # Using gr.Label handles classification UI colors beautifully automatically
            output_label = gr.Label(label="Diagnostic Results")
            
            # Action Bindings
            sample_btn.click(fn=lambda: SAMPLE_PATIENT, inputs=None, outputs=inputs_list)
            clear_btn.click(fn=lambda: [0.0]*30, inputs=None, outputs=inputs_list)
            predict_btn.click(fn=predict_single, inputs=inputs_list, outputs=output_label)
            
        # TAB 2: Batch CSV Processing UI
        with gr.TabItem("Batch Prediction (CSV & PDF)"):
            gr.Markdown("### 📂 Bulk Classification Processing System")
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
                    batch_btn = gr.Button("Process Automation Data", variant="primary")
                with gr.Column():
                    status_txt = gr.Textbox(label="System Status", value="Awaiting file upload workflow stream.", interactive=False)
                    
                    # Group containing clean numerical statistical boxes
                    with gr.Group():
                        gr.Markdown("#### 📊 Batch Processing Statistics")
                        with gr.Row():
                            stat_total = gr.Number(label="Total Evaluated", value=0, precision=0, interactive=False)
                            stat_benign = gr.Number(label="Benign Cases", value=0, precision=0, interactive=False)
                            stat_malignant = gr.Number(label="Malignant Cases", value=0, precision=0, interactive=False)
                            stat_conf = gr.Number(label="Avg Confidence (%)", value=0.0, precision=2, interactive=False)
                            
                    with gr.Row():
                        csv_output = gr.File(label="Download Enriched CSV")
                        pdf_output = gr.File(label="Download PDF Report")
                        
            batch_btn.click(
                fn=predict_batch, 
                inputs=file_input, 
                outputs=[csv_output, pdf_output, status_txt, stat_total, stat_benign, stat_malignant, stat_conf]
            )

    gr.Markdown("--- \nDeveloped by Abhay Dwivedi | **Machine Learning Portfolio Project**")

if __name__ == "__main__":
    demo.queue().launch()