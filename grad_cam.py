import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for the given image"""
    try:
        # Create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(
                last_conv_layer_name).output, model.output]
        )

        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron with regard to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Vector where each entry is the mean intensity of the gradient over a feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map array by its importance
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        raise Exception(f"Grad-CAM heatmap generation failed: {str(e)}")


def get_conv_layer_names(model):
    """Get all convolutional layer names in the model"""
    conv_layers = []
    for layer in model.layers:
        if 'conv' in layer.name:
            conv_layers.append(layer.name)
    return conv_layers


def generate_gradcam(model, img_array, original_img):
    """Generate Grad-CAM visualization for the given image"""
    try:
        print("🔍 Looking for convolutional layers...")

        # Get all convolutional layers
        conv_layers = get_conv_layer_names(model)
        print(f"✅ Found convolutional layers: {conv_layers}")

        if not conv_layers:
            raise Exception("No convolutional layers found in the model")

        # Use the LAST convolutional layer (usually gives the best results)
        last_conv_layer_name = conv_layers[-1]
        print(f"🎯 Using layer: {last_conv_layer_name}")

        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        print(f"📐 Original image shape: {original_img.shape}")
        print(f"📐 Heatmap shape: {heatmap.shape}")

        # Resize heatmap to match original image
        heatmap = cv2.resize(
            heatmap, (original_img.shape[1], original_img.shape[0]))

        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        print("🎨 Applying heatmap to image...")

        # Ensure both images have the same data type and range
        if original_img.dtype != np.uint8:
            original_img = (original_img * 255).astype(np.uint8)

        # If original image is grayscale, convert to RGB
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif original_img.shape[2] == 1:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)

        print("✅ Grad-CAM successfully generated!")
        return heatmap, superimposed_img

    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
        # Return the original image as fallback
        return None, original_img
