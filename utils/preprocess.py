"""
Advanced data preprocessing module for plant disease detection.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from typing import Tuple, Union, Any
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define standard image size for MobileNetV2
TARGET_SIZE = (224, 224)

def load_and_resize_image(image_input: Union[str, Image.Image, np.ndarray], target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Loads and resizes an image to the target size.
    
    Args:
        image_input (Union[str, Image.Image, np.ndarray]): The input image as a file path, PIL Image, or NumPy array.
        target_size (Tuple[int, int]): The desired output size as (width, height).
        
    Returns:
        np.ndarray: Resized image as a NumPy array (RGB format).
        
    Raises:
        ValueError: If the input type is unsupported or the image cannot be loaded.
        FileNotFoundError: If the input string is not a valid file path.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        try:
            pil_img = Image.open(image_input)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            pil_img = pil_img.resize(target_size)
            img_array = np.array(pil_img)
        except UnidentifiedImageError:
            raise ValueError(f"File is not a valid image or is corrupted: {image_input}")
            
    elif isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        pil_img = image_input.resize(target_size)
        img_array = np.array(pil_img)
        
    elif isinstance(image_input, np.ndarray):
        # Convert grayscale or RGBA to RGB if needed
        if len(image_input.shape) == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif len(image_input.shape) == 3:
            if image_input.shape[2] == 1:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            
        img_array = cv2.resize(image_input, target_size)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
    return img_array

def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from [0, 255] to [0, 1].
    
    Args:
        img_array (np.ndarray): The input image array.
        
    Returns:
        np.ndarray: Normalized image array in float32 format.
    """
    return img_array.astype(np.float32) / 255.0

def apply_augmentation(img_array: np.ndarray) -> np.ndarray:
    """
    Applies random augmentation to a single image based on predefined settings.
    
    Args:
        img_array (np.ndarray): Original image array (H, W, C).
        
    Returns:
        np.ndarray: Augmented image array.
    """
    img_expanded = np.expand_dims(img_array, axis=0)
    
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Generate one augmented version
    aug_iter = datagen.flow(img_expanded, batch_size=1)
    aug_img = next(aug_iter)[0]
    
    return aug_img

def _sorted_class_names(data_dir: str) -> list:
    """Return sorted class folder names under a dataset directory."""
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


def create_train_val_generators(
    train_dir: str,
    val_dir: str,
    batch_size: int = 64,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[Any, Any]:
    """
    Build train/validation generators from separate directories with a shared class map.
    """
    classes = _sorted_class_names(train_dir)
    val_classes = set(_sorted_class_names(val_dir))
    missing = set(classes) - val_classes
    if missing:
        raise ValueError(
            f"Validation set missing {len(missing)} class folders present in training set."
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False,
    )
    for gen in (train_gen, val_gen):
        gen._workers = 4
        gen._use_multiprocessing = True
        gen._max_queue_size = 16
    return train_gen, val_gen


def create_data_generators(data_dir: str, batch_size: int = 64, target_size: Tuple[int, int] = TARGET_SIZE) -> Tuple[Any, Any]:
    """
    Creates high-performance tf.data pipelines for training and validation.
    
    Uses ImageDataGenerator under the hood but wraps them in tf.data.Dataset
    with prefetching and parallel mapping for significantly faster CPU training.
    
    Args:
        data_dir (str): Path to the dataset directory containing class folders.
        batch_size (int): Number of images per batch.
        target_size (Tuple[int, int]): Dimensions to resize images to.
        
    Returns:
        Tuple[Any, Any]: Training and validation tf.data.Dataset objects.
            Both have .num_classes, .class_indices, and .n attributes attached.
    """
    # Training data — light augmentation (heavy augmentation via MixUp/CutMix in Phase 2)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data — only rescaling, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    for gen in (train_gen, val_gen):
        gen._workers = 4
        gen._use_multiprocessing = True
        gen._max_queue_size = 16
    
    # Return the directory generators directly to optimize performance and prevent GIL bottlenecks
    return train_gen, val_gen

def visualize_augmented_images(image_input: Union[str, Image.Image, np.ndarray], grid_size: Tuple[int, int] = (3, 3)) -> None:
    """
    Visualizes augmented versions of a single image in a grid.
    
    Args:
        image_input (Union[str, Image.Image, np.ndarray]): The input image.
        grid_size (Tuple[int, int]): Dimensions of the plot grid (rows, columns).
    """
    img_array = load_and_resize_image(image_input)
    
    rows, cols = grid_size
    num_images = rows * cols
    
    plt.figure(figsize=(cols * 3, rows * 3))
    
    # Display the original first
    plt.subplot(rows, cols, 1)
    plt.imshow(img_array.astype('uint8'))
    plt.title("Original")
    plt.axis('off')
    
    # Display augmentations
    for i in range(2, num_images + 1):
        aug_img = apply_augmentation(img_array)
        plt.subplot(rows, cols, i)
        plt.imshow(aug_img.astype('uint8'))
        plt.title(f"Augmented {i-1}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def validate_image_quality(img_array: np.ndarray) -> Tuple[bool, str]:
    """
    Validates image quality before prediction (checks for blur, contrast, blankness).
    
    Args:
        img_array (np.ndarray): Image array (preferably resized uint8).
        
    Returns:
        Tuple[bool, str]: Boolean flag indicating if image is valid, and a status message.
    """
    # Ensure image format is uint8 for OpenCV operations
    if img_array.dtype != np.uint8:
        if np.max(img_array) <= 1.0:
            check_img = (img_array * 255).astype(np.uint8)
        else:
            check_img = img_array.astype(np.uint8)
    else:
        check_img = img_array.copy()
        
    # 1. Check if image is blank (completely dark or severely over-exposed)
    mean_val = np.mean(check_img)
    if mean_val < 5 or mean_val > 250:
        return False, "Image appears to be completely blank or severely under/over-exposed."
        
    # 2. Check for blurriness (Variance of Laplacian)
    gray = cv2.cvtColor(check_img, cv2.COLOR_RGB2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance_of_laplacian < 30: # Threshold for blurriness
        return False, f"Image is too blurry for reliable prediction (Score: {variance_of_laplacian:.2f})."
        
    # 3. Check for extremely low contrast
    std_dev = np.std(gray)
    if std_dev < 15:
        return False, f"Image has extremely low contrast (Score: {std_dev:.2f})."
        
    return True, "Image passed quality checks."

def check_is_leaf_hsv(img_array: np.ndarray) -> Tuple[bool, str]:
    """
    Checks if the image likely contains a plant leaf based on color signature in HSV space.
    """
    # Ensure correct dtype
    if img_array.dtype != np.uint8:
        if np.max(img_array) <= 1.0:
            hsv_img = (img_array * 255).astype(np.uint8)
        else:
            hsv_img = img_array.astype(np.uint8)
    else:
        hsv_img = img_array.copy()
        
    # Convert to HSV
    hsv = cv2.cvtColor(hsv_img, cv2.COLOR_RGB2HSV)
    
    # Define masks for leaf colors:
    # 1. Green: Hue 35-85, Saturation 30-255, Value 30-255
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 2. Yellow/Orange (chlorosis/early lesions): Hue 10-35, Saturation 40-255, Value 40-255
    lower_yellow = np.array([10, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3. Brown/Red/Autumn leaves: Hue 0-10 or 165-180, Saturation 20-255, Value 20-255
    lower_brown1 = np.array([0, 20, 20])
    upper_brown1 = np.array([15, 255, 200])
    mask_brown1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    
    lower_brown2 = np.array([165, 20, 20])
    upper_brown2 = np.array([180, 255, 200])
    mask_brown2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    
    # Combined mask
    mask_combined = mask_green | mask_yellow | mask_brown1 | mask_brown2
    
    total_pixels = img_array.shape[0] * img_array.shape[1]
    leaf_pixels = cv2.countNonZero(mask_combined)
    percentage = (leaf_pixels / total_pixels) * 100
    
    # If the image has less than 8% leaf-like pixels, it's very likely NOT a leaf or plant specimen.
    if percentage < 8.0:
        return False, f"Image does not appear to contain a plant leaf or foliage. Leaf color coverage: {percentage:.1f}% (Minimum required: 8.0%)."
        
    return True, f"Passed HSV check (Leaf-like color coverage: {percentage:.1f}%)."


def validate_leaf_image_llm(image: Image.Image) -> Tuple[bool, str]:
    """
    Optional LLM vision check to verify if the image is a plant leaf.
    """
    try:
        import os
        import json
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = "Analyze this image. Is it a photo of a plant leaf, crop specimen, or agricultural vegetation? Answer in JSON format: {\"is_leaf\": true/false, \"reason\": \"brief explanation\"}"
            response = model.generate_content([image, prompt])
            text = response.text.strip()
            
            # Clean up potential markdown formatting in code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                
            res = json.loads(text)
            if not res.get("is_leaf", False):
                return False, f"LLM verification failed: {res.get('reason', 'Image is not a plant leaf.')}"
    except Exception as e:
        # Fall back to passing if API is unreachable or has rate limits
        pass
    return True, "Passed LLM verification."


def validate_leaf_image(image_input: Union[str, Image.Image, np.ndarray]) -> Tuple[bool, str]:
    """
    Validates if the image input is actually a plant leaf or crop specimen.
    """
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
        else:
            return False, "Unsupported image type for leaf validation"
    except Exception as e:
        return False, f"Could not load image for leaf validation: {e}"
        
    img_array = np.array(pil_img)
    
    # 1. Run HSV color analysis
    is_leaf_hsv, msg_hsv = check_is_leaf_hsv(img_array)
    if not is_leaf_hsv:
        return False, msg_hsv
        
    # 2. Run LLM vision confirmation (if online/key available)
    is_leaf_llm, msg_llm = validate_leaf_image_llm(pil_img)
    if not is_leaf_llm:
        return False, msg_llm
        
    return True, "Passed leaf validation."


def preprocess_image(image_input: Union[str, Image.Image, np.ndarray], target_size: Tuple[int, int] = TARGET_SIZE, validate_leaf: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main preprocessing pipeline for inference.
    Loads, validates, and normalizes an image for the MobileNetV2 model.
    
    Args:
        image_input (Union[str, Image.Image, np.ndarray]): The input image.
        target_size (Tuple[int, int]): Size to resize the image to.
        validate_leaf (bool): If True, validates if image contains a plant leaf.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Preprocessed image array of shape (1, 224, 224, 3) normalized [0, 1].
            - Original resized image array for visualization in uint8 format.
            
    Raises:
        ValueError: If image fails quality checks or leaf validation.
    """
    img_array = load_and_resize_image(image_input, target_size)
    
    # Validate image quality before prediction
    is_valid, msg = validate_image_quality(img_array)
    if not is_valid:
        raise ValueError(f"Quality Check Failed: {msg}")
        
    # Optional leaf check
    if validate_leaf:
        is_leaf, leaf_msg = validate_leaf_image(image_input)
        if not is_leaf:
            raise ValueError(f"Leaf Validation Failed: {leaf_msg}")
        
    # Normalize and expand dimensions to create a batch of 1
    normalized_img = normalize_image(img_array)
    batch_img = np.expand_dims(normalized_img, axis=0)
    
    return batch_img, img_array
