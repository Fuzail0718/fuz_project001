import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import grad_cam
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .normal {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .pneumonia {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .confidence-high {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-low {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    .test-pass {
        color: #28a745;
        font-weight: bold;
    }
    .test-fail {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pneumonia_model.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image):
    """Preprocess the uploaded image for the model"""
    # Resize to match model input
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Normalize pixel values
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_model_validation(model):
    """Run comprehensive model validation"""
    results = {
        'pneumonia_tests': [],
        'normal_tests': [],
        'summary': {'total_tests': 0, 'passed_tests': 0}
    }

    # Test pneumonia images
    pneumonia_dir = "data/test/PNEUMONIA"
    if os.path.exists(pneumonia_dir):
        pneumonia_files = [f for f in os.listdir(
            pneumonia_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_files = pneumonia_files[:3]  # Test first 3 images

        for file in test_files:
            try:
                image_path = os.path.join(pneumonia_dir, file)
                image = Image.open(image_path)
                processed = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)[0][0]

                # Pneumonia should have prediction > 0.5
                passed = prediction > 0.5
                results['pneumonia_tests'].append({
                    'file': file,
                    'prediction': prediction,
                    'passed': passed,
                    'expected': 'PNEUMONIA'
                })

                if passed:
                    results['summary']['passed_tests'] += 1
                results['summary']['total_tests'] += 1

            except Exception as e:
                st.error(f"Error testing {file}: {e}")

    # Test normal images
    normal_dir = "data/test/NORMAL"
    if os.path.exists(normal_dir):
        normal_files = [f for f in os.listdir(
            normal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_files = normal_files[:3]  # Test first 3 images

        for file in test_files:
            try:
                image_path = os.path.join(normal_dir, file)
                image = Image.open(image_path)
                processed = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)[0][0]

                # Normal should have prediction < 0.5
                passed = prediction < 0.5
                results['normal_tests'].append({
                    'file': file,
                    'prediction': prediction,
                    'passed': passed,
                    'expected': 'NORMAL'
                })

                if passed:
                    results['summary']['passed_tests'] += 1
                results['summary']['total_tests'] += 1

            except Exception as e:
                st.error(f"Error testing {file}: {e}")

    return results


def display_confidence_metrics(pneumonia_prob, normal_prob):
    """Display confidence metrics and gauges"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pneumonia Probability", f"{pneumonia_prob:.2%}")
        st.progress(float(pneumonia_prob))

    with col2:
        st.metric("Normal Probability", f"{normal_prob:.2%}")
        st.progress(float(normal_prob))

    with col3:
        confidence_level = ""
        if max(pneumonia_prob, normal_prob) > 0.8:
            confidence_level = "🟢 High Confidence"
        elif max(pneumonia_prob, normal_prob) > 0.6:
            confidence_level = "🟡 Medium Confidence"
        else:
            confidence_level = "🔴 Low Confidence"

        st.metric("Confidence Level", confidence_level)


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 AI-Powered Pneumonia Detection</h1>',
                unsafe_allow_html=True)
    st.markdown("### Upload a chest X-ray image for analysis")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This AI system helps detect pneumonia from chest X-ray images using deep learning. "
        "It highlights areas of concern using Grad-CAM visualization to provide explainable results."
    )

    st.sidebar.title("Instructions")
    st.sidebar.write("""
    1. Upload a chest X-ray image (JPEG/PNG)
    2. Click 'Analyze Image'
    3. View the AI prediction and heatmap
    4. Download the report if needed
    """)

    st.sidebar.title("Model Validation")

    model = load_model()

    if model is not None and st.sidebar.button("🧪 Run Model Validation"):
        with st.sidebar:
            st.info("Running comprehensive validation tests...")
            validation_results = run_model_validation(model)

            st.write("### Validation Results:")

            # Display pneumonia test results
            if validation_results['pneumonia_tests']:
                st.write("**Pneumonia Images Test:**")
                for test in validation_results['pneumonia_tests']:
                    status = "✅ PASS" if test['passed'] else "❌ FAIL"
                    color_class = "test-pass" if test['passed'] else "test-fail"
                    st.markdown(
                        f"<span class='{color_class}'>{status}</span> {test['file']}: {test['prediction']:.3f}", unsafe_allow_html=True)

            # Display normal test results
            if validation_results['normal_tests']:
                st.write("**Normal Images Test:**")
                for test in validation_results['normal_tests']:
                    status = "✅ PASS" if test['passed'] else "❌ FAIL"
                    color_class = "test-pass" if test['passed'] else "test-fail"
                    st.markdown(
                        f"<span class='{color_class}'>{status}</span> {test['file']}: {test['prediction']:.3f}", unsafe_allow_html=True)

            # Summary
            total = validation_results['summary']['total_tests']
            passed = validation_results['summary']['passed_tests']
            accuracy = passed / total if total > 0 else 0

            st.write("### Summary:")
            st.write(f"**Tests Passed:** {passed}/{total} ({accuracy:.1%})")

            if accuracy >= 0.8:
                st.success("🎉 Model is performing well!")
            elif accuracy >= 0.6:
                st.warning("⚠️ Model performance needs improvement")
            else:
                st.error("❌ Model needs retraining")

    st.sidebar.title("Model Information")
    st.sidebar.write("""
    - **Model Type**: Convolutional Neural Network
    - **Training Data**: Chest X-ray images
    - **Classes**: Normal vs Pneumonia
    - **Features**: Explainable AI with Grad-CAM
    """)

    st.sidebar.title("Performance Tips")
    st.sidebar.write("""
    - Use clear, frontal chest X-rays
    - Ensure good image quality
    - Test with known cases first
    - Check validation results above
    """)

    st.sidebar.title("Disclaimer")
    st.sidebar.warning(
        "This tool is for educational and research purposes only. "
        "Always consult healthcare professionals for medical diagnosis."
    )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a frontal chest X-ray image"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing the image..."):
                    # Load model
                    model = load_model()

                    if model is not None:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        original_image = np.array(image)

                        # Make prediction
                        prediction = model.predict(
                            processed_image, verbose=0)[0][0]
                        pneumonia_prob = float(prediction)
                        normal_prob = 1 - pneumonia_prob

                        # Determine result
                        if pneumonia_prob > 0.5:
                            result = "PNEUMONIA"
                            confidence = pneumonia_prob
                        else:
                            result = "NORMAL"
                            confidence = normal_prob

                        # Generate Grad-CAM
                        heatmap_available = False
                        superimposed_img = None

                        try:
                            st.info("🔄 Generating AI attention map...")

                            # Add debug info
                            with st.expander("Model Details"):
                                st.write(
                                    f"**Model Layers:** {[layer.name for layer in model.layers]}")

                            heatmap, superimposed_img = grad_cam.generate_gradcam(
                                model, processed_image, original_image
                            )

                            if superimposed_img is not None and not np.array_equal(superimposed_img, original_image):
                                st.success("✅ AI attention map generated!")
                                heatmap_available = True
                            else:
                                st.warning("⚠️ Using fallback visualization")
                                heatmap_available = False

                        except Exception as e:
                            st.warning(
                                f"⚠️ Grad-CAM visualization failed: {e}")
                            heatmap_available = False

                        # Display results
                        with col2:
                            st.subheader("Analysis Results")

                            # Prediction box
                            if result == "PNEUMONIA":
                                st.markdown(
                                    f'<div class="result-box pneumonia">'
                                    f'<h3>🚨 Pneumonia Detected</h3>'
                                    f'<p>Confidence: <span class="confidence-high">{confidence:.2%}</span></p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                                st.error(
                                    "⚠️ This result suggests possible pneumonia. Please consult a healthcare professional.")
                            else:
                                st.markdown(
                                    f'<div class="result-box normal">'
                                    f'<h3>✅ Normal Chest X-ray</h3>'
                                    f'<p>Confidence: <span class="confidence-low">{confidence:.2%}</span></p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                                st.success("✅ No signs of pneumonia detected.")

                            # Display confidence metrics
                            st.subheader("Confidence Metrics")
                            display_confidence_metrics(
                                pneumonia_prob, normal_prob)

                            # Detailed probabilities
                            with st.expander("Detailed Probabilities"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Pneumonia Probability",
                                              f"{pneumonia_prob:.4f}")
                                with col_b:
                                    st.metric("Normal Probability",
                                              f"{normal_prob:.4f}")

                            # Visualization
                            if heatmap_available and superimposed_img is not None:
                                st.subheader("AI Attention Map")
                                st.write(
                                    "The heatmap shows where the AI is looking to make its decision (red areas = more attention):")

                                fig, (ax1, ax2) = plt.subplots(
                                    1, 2, figsize=(12, 5))

                                # Original image
                                ax1.imshow(original_image)
                                ax1.set_title("Original X-ray")
                                ax1.axis('off')

                                # Heatmap
                                ax2.imshow(cv2.cvtColor(
                                    superimposed_img, cv2.COLOR_BGR2RGB))
                                ax2.set_title("AI Attention Heatmap")
                                ax2.axis('off')

                                st.pyplot(fig)
                            else:
                                # Fallback visualization
                                st.subheader("Probability Distribution")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                categories = ['Normal', 'Pneumonia']
                                probabilities = [normal_prob, pneumonia_prob]
                                colors = ['#28a745', '#dc3545']

                                bars = ax.bar(
                                    categories, probabilities, color=colors, alpha=0.7)
                                ax.set_ylim(0, 1)
                                ax.set_ylabel('Probability')
                                ax.set_title('AI Diagnosis Confidence')

                                # Add value labels on bars
                                for bar, prob in zip(bars, probabilities):
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                            f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

                                st.pyplot(fig)

                            # Download report
                            st.subheader("Download Report")
                            report_text = f"""
PNEUMONIA DETECTION REPORT
=========================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
File Name: {uploaded_file.name}

RESULTS:
- Prediction: {result}
- Confidence: {confidence:.2%}
- Pneumonia Probability: {pneumonia_prob:.4f}
- Normal Probability: {normal_prob:.4f}

MODEL INFORMATION:
- AI Model: Convolutional Neural Network
- Training: Chest X-ray dataset
- Purpose: Educational/Research

IMPORTANT DISCLAIMER:
This report is generated by an AI system for educational and research purposes only.
This is NOT a medical diagnosis. Always consult qualified healthcare professionals 
for medical advice and diagnosis.

Generated by: AI Pneumonia Detection System
                            """

                            st.download_button(
                                label="📄 Download Text Report",
                                data=report_text,
                                file_name=f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )

    # If no file uploaded, show sample workflow
    if uploaded_file is None:
        with col2:
            st.info("👆 Upload a chest X-ray image to get started")
            st.image("https://via.placeholder.com/400x300/4DA6FF/FFFFFF?text=Upload+X-ray+Image",
                     caption="Sample workflow: Upload → Analyze → View Results")


if __name__ == "__main__":
    main()
