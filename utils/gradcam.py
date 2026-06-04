"""
Advanced Grad-CAM Explainability Module for Plant Disease Detection.
Provides automatic conv layer detection, heatmap generation, overlay blending, 
and visualization figures.
"""

import logging
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Automatically detects the name of the last convolutional layer in a model.
    Handles nested architectures, MobileNetV2, and EfficientNet.
    
    Args:
        model (tf.keras.Model): The trained Keras model.
        
    Returns:
        str: The name of the last convolutional layer.
        
    Raises:
        ValueError: If no convolutional layer could be automatically detected.
    """
    try:
        # Search backward through layers
        for layer in reversed(model.layers):
            # Check if layer is a Conv layer
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return layer.name
                
            # If the model itself contains a nested sub-model (like MobileNetV2 base)
            if hasattr(layer, 'layers') and isinstance(layer, tf.keras.Model):
                # Search backward inside the sub-model
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        return sub_layer.name
                        
            # Specific check for MobileNetV2 and EfficientNet output ReLU layers (the layer right before pooling)
            if layer.name == 'out_relu' or 'top_conv' in layer.name:
                return layer.name
                
        # Second pass: check sub-models layer names directly if they are treated as standalone
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):
                try:
                    sub_name = get_last_conv_layer_name(layer)
                    return sub_name
                except ValueError:
                    continue
                    
        # Third pass: string name matching fallback
        for layer in reversed(model.layers):
            if any(term in layer.name.lower() for term in ['conv', 'relu', 'block']):
                return layer.name
                
        raise ValueError("No convolutional layer detected in the model architecture.")
    except Exception as e:
        logger.error(f"Error in get_last_conv_layer_name: {str(e)}")
        raise

def generate_heatmap(model: tf.keras.Model, img_array: np.ndarray, layer_name: str) -> np.ndarray:
    """
    Generates a raw Grad-CAM heatmap for a given image and model.
    
    Args:
        model (tf.keras.Model): The trained Keras model.
        img_array (np.ndarray): Preprocessed image array of shape (1, H, W, C).
        layer_name (str): Name of the convolutional layer to use for activation maps.
        
    Returns:
        np.ndarray: Normalized 2D heatmap of shape (H, W).
    """
    try:
        # Find the target layer, checking nested models if necessary
        target_layer = None
        
        if layer_name in [l.name for l in model.layers]:
            target_layer = model.get_layer(layer_name)
        else:
            # Check if target layer is in one of the nested models
            for layer in model.layers:
                if hasattr(layer, 'layers') and isinstance(layer, tf.keras.Model):
                    if layer_name in [l.name for l in layer.layers]:
                        target_layer = layer.get_layer(layer_name)
                        break
                        
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model or sub-models.")

        # Construct a gradient model that maps main inputs to inner layer activations AND final predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [target_layer.output, model.output]
        )
            
        # TODO: Generate Grad-CAM heatmap using GradientTape for last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            best_class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, best_class_idx]
            
        # Gradients of the top class score w.r.t. conv outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Mean intensity of gradients over each channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted sum of channel activations
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU of heatmap, normalized [0, 1]
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            max_val = 1e-10  # Prevent divide by zero
        heatmap = heatmap / max_val
        
        return heatmap.numpy()
        
    except Exception as e:
        logger.error(f"Error in generate_heatmap: {str(e)}")
        raise

def overlay_heatmap_on_image(original_image: Union[Image.Image, np.ndarray], heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """
    Superimposes a Grad-CAM heatmap over the original image using OpenCV JET colormap.
    
    Args:
        original_image (Union[Image.Image, np.ndarray]): The source image.
        heatmap (np.ndarray): 2D Grad-CAM heatmap array [0, 1].
        alpha (float): Transparency multiplier for the heatmap overlay.
        
    Returns:
        Image.Image: The blended PIL Image.
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(original_image, Image.Image):
            orig_img = np.array(original_image)
        else:
            orig_img = original_image.copy()
            
        # Ensure correct datatype uint8
        if orig_img.dtype != np.uint8:
            if np.max(orig_img) <= 1.0:
                orig_img = (orig_img * 255).astype(np.uint8)
            else:
                orig_img = orig_img.astype(np.uint8)

        # Scale heatmap to [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # TODO: Overlay heatmap on original image using cv2 colormap and alpha blend
        # Apply JET colormap using OpenCV
        colormap_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Resize colormap heatmap to match original image dimensions
        colormap_heatmap = cv2.resize(colormap_heatmap, (orig_img.shape[1], orig_img.shape[0]))
        
        # OpenCV colormap is BGR, convert to RGB
        colormap_heatmap = cv2.cvtColor(colormap_heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend the heatmap and original image
        blended = cv2.addWeighted(orig_img, 1.0 - alpha, colormap_heatmap, alpha, 0)
        
        return Image.fromarray(blended)
        
    except Exception as e:
        logger.error(f"Error in overlay_heatmap_on_image: {str(e)}")
        raise

def create_gradcam_comparison(model: tf.keras.Model, image: Union[str, Image.Image, np.ndarray], img_size: int = 224) -> Dict[str, Any]:
    """
    Full pipeline to generate Grad-CAM original, raw heatmap, and overlay images.
    
    Args:
        model (tf.keras.Model): Keras model.
        image (Union[str, Image.Image, np.ndarray]): Source image file path, PIL Image, or numpy array.
        img_size (int): Target dimensions.
        
    Returns:
        Dict[str, Any]: Dictionary containing comparison results and execution status.
    """
    result = {
        "original": None,
        "heatmap": None,
        "overlay": None,
        "success": False,
        "error_message": ""
    }
    
    try:
        from utils.preprocess import load_and_resize_image, normalize_image
        
        # 1. Load and resize
        orig_array = load_and_resize_image(image, (img_size, img_size))
        result["original"] = Image.fromarray(orig_array)
        
        # 2. Preprocess for model input
        norm_img = normalize_image(orig_array)
        batch_img = np.expand_dims(norm_img, axis=0)
        
        # 3. Detect layer
        layer_name = get_last_conv_layer_name(model)
        
        # 4. Generate Heatmap
        heatmap = generate_heatmap(model, batch_img, layer_name)
        result["heatmap"] = heatmap
        
        # 5. Overlay Heatmap
        overlay_img = overlay_heatmap_on_image(orig_array, heatmap)
        result["overlay"] = overlay_img
        
        result["success"] = True
        logger.info("Grad-CAM comparison successfully generated.")
        
    except Exception as e:
        err_msg = f"Failed to complete Grad-CAM pipeline: {str(e)}"
        logger.error(err_msg)
        result["error_message"] = err_msg
        
    return result

def create_side_by_side_figure(original: Union[Image.Image, np.ndarray], overlay: Union[Image.Image, np.ndarray]) -> plt.Figure:
    """
    Creates a matplotlib figure displaying the original image and overlay side-by-side with a colorbar.
    
    Args:
        original (Union[Image.Image, np.ndarray]): Original image.
        overlay (Union[Image.Image, np.ndarray]): Heatmap-overlayed image.
        
    Returns:
        plt.Figure: Matplotlib figure object.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Show original
        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Show overlay
        axes[1].imshow(overlay)
        axes[1].set_title("AI Attention Map (Grad-CAM)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar for mapping reference
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity (Blue -> Red)', fontsize=10)
        cbar.set_ticks([]) # Clean styling
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error in create_side_by_side_figure: {str(e)}")
        raise

# --- BACKWARD COMPATIBILITY ALIASES ---
def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, pred_index: int = None) -> np.ndarray:
    """
    Wrapper function for backward compatibility with app.py.
    """
    return generate_heatmap(model, img_array, last_conv_layer_name)

def display_gradcam(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Wrapper function for backward compatibility with app.py.
    """
    pil_img = overlay_heatmap_on_image(img_array, heatmap, alpha)
    return np.array(pil_img)
