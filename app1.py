from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
model = None


def load_model():
    global model
    try:
        model = tf.keras.models.load_model('pneumonia_model.h5')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


load_model()


def preprocess_image(image):
    """Preprocess uploaded image for model"""
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def generate_heatmap_visualization(original_img, pneumonia_prob):
    """Generate a professional-looking visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    ax1.imshow(original_img)
    ax1.set_title('Uploaded X-ray', fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')

    # Risk indicator
    risk_level = "HIGH RISK" if pneumonia_prob > 0.5 else "LOW RISK"
    risk_color = '#dc3545' if pneumonia_prob > 0.5 else '#28a745'

    ax2.barh(['Pneumonia', 'Normal'],
             [pneumonia_prob, 1 - pneumonia_prob],
             color=[risk_color, '#6c757d'])
    ax2.set_xlim(0, 1)
    ax2.set_title('Diagnostic Confidence', fontsize=14,
                  fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate([pneumonia_prob, 1 - pneumonia_prob]):
        ax2.text(v + 0.01, i, f'{v:.1%}', color='black', fontweight='bold')

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Read and process image
        image = Image.open(file.stream)
        processed_img = preprocess_image(image)
        original_img = np.array(image.resize((224, 224)))

        # Make prediction
        prediction = model.predict(processed_img, verbose=0)[0][0]
        pneumonia_prob = float(prediction)
        normal_prob = 1 - pneumonia_prob

        # Determine result
        if pneumonia_prob > 0.5:
            result = "PNEUMONIA_DETECTED"
            status = "high-risk"
            recommendation = "Immediate medical consultation recommended"
            icon = "⚠️"
        else:
            result = "NORMAL"
            status = "low-risk"
            recommendation = "No immediate concerns detected"
            icon = "✅"

        # Generate visualization
        visualization = generate_heatmap_visualization(
            original_img, pneumonia_prob)

        response = {
            'success': True,
            'result': result,
            'status': status,
            'icon': icon,
            'probabilities': {
                'pneumonia': pneumonia_prob,
                'normal': normal_prob
            },
            'confidence': max(pneumonia_prob, normal_prob),
            'recommendation': recommendation,
            'visualization': f"data:image/png;base64,{visualization}",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': file.filename
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/report', methods=['POST'])
def generate_report():
    data = request.json
    report_text = f"""
MEDICAL IMAGE ANALYSIS REPORT
=============================

Patient Study: Chest X-ray Analysis
Generated: {data.get('timestamp', 'N/A')}
File: {data.get('filename', 'N/A')}

DIAGNOSTIC RESULTS:
------------------
Result: {data.get('result', 'N/A').replace('_', ' ').title()}
Confidence Level: {data.get('confidence', 0):.2%}

Detailed Probabilities:
- Pneumonia: {data.get('probabilities', {}).get('pneumonia', 0):.4f}
- Normal: {data.get('probabilities', {}).get('normal', 0):.4f}

CLINICAL RECOMMENDATION:
-----------------------
{data.get('recommendation', 'N/A')}

IMPORTANT DISCLAIMER:
--------------------
This report is generated by an AI system for educational and research purposes only.
This is NOT a medical diagnosis. The results should be interpreted by qualified 
healthcare professionals. Always consult with medical experts for proper diagnosis 
and treatment planning.

AI Pneumonia Detection System
Medical Imaging Analysis Module
    """

    return jsonify({'report': report_text})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
