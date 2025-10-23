import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set page config FIRST
st.set_page_config(
    page_title="AI Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 15px;
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
</style>
""", unsafe_allow_html=True)

# Simple Grad-CAM implementation
def simple_gradcam(model, img_array, original_img):
    """Simplified Grad-CAM for testing"""
    try:
        # For simple CNN models
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            return None, original_img
            
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_output, pooled_grads), axis=-1)
        
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        return heatmap, superimposed_img
        
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        return None, original_img

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        if os.path.exists('pneumonia_model.h5'):
            model = tf.keras.models.load_model('pneumonia_model.h5')
            return model
        else:
            st.error("Model file 'pneumonia_model.h5' not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    elif img_array.shape[2] == 1:  # Single channel
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.markdown('<h1 class="main-header">🫁 AI Pneumonia Detection</h1>', unsafe_allow_html=True)
    st.write("Upload a chest X-ray image for analysis")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("This AI helps detect pneumonia from chest X-rays using deep learning.")
    
    st.sidebar.title("Instructions")
    st.sidebar.write("1. Upload X-ray image\n2. Click Analyze\n3. View results")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose X-ray image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    model = load_model()
                    
                    if model is not None:
                        # Preprocess and predict
                        processed_img = preprocess_image(image)
                        original_img = np.array(image)
                        
                        prediction = model.predict(processed_img, verbose=0)[0][0]
                        pneumonia_prob = float(prediction)
                        normal_prob = 1 - pneumonia_prob
                        
                        if pneumonia_prob > 0.5:
                            result = "PNEUMONIA"
                            confidence = pneumonia_prob
                            css_class = "pneumonia"
                        else:
                            result = "NORMAL"
                            confidence = normal_prob
                            css_class = "normal"
                        
                        # Display results
                        with col2:
                            st.subheader("Results")
                            st.markdown(
                                f'<div class="result-box {css_class}">'
                                f'<h3>{"🚨 Pneumonia" if result == "PNEUMONIA" else "✅ Normal"}</h3>'
                                f'<p>Confidence: <strong>{confidence:.2%}</strong></p>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )
                            
                            # Try Grad-CAM
                            heatmap, superimposed = simple_gradcam(model, processed_img, original_img)
                            if superimposed is not None:
                                st.subheader("AI Attention Map")
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                                ax1.imshow(original_img)
                                ax1.set_title("Original")
                                ax1.axis('off')
                                ax2.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                                ax2.set_title("AI Focus Areas")
                                ax2.axis('off')
                                st.pyplot(fig)
                            
                            # Report
                            st.subheader("Report")
                            report = f"""
                            Pneumonia Detection Report
                            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                            
                            Result: {result}
                            Confidence: {confidence:.2%}
                            Pneumonia Probability: {pneumonia_prob:.4f}
                            
                            Note: Educational use only. Consult a doctor.
                            """
                            st.download_button(
                                "📄 Download Report",
                                report,
                                f"report_{datetime.now().strftime('%H%M%S')}.txt"
                            )

if __name__ == "__main__":
    main()