"""
Model Building Module for Plant Disease Detection.
Supports MobileNetV2 and EfficientNetB0 architectures with custom classification heads.
Includes utilities for fine-tuning, summarization, and saving models in multiple formats.
"""

import os
import logging
from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, EfficientNetB3, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tensorflow.keras.regularizers import l2

def _build_custom_head(
    base_model: Model, 
    num_classes: int,
    dense_units_1: int = 512,
    dense_units_2: int = 256,
    dropout_rate_1: float = 0.5,
    dropout_rate_2: float = 0.3,
    l2_reg: float = 0.001,
    preprocessing_layer = None
) -> Model:
    """
    Helper function to build a custom classification head on top of a base model.
    Allows adjusting dense units, dropouts, and L2 weight regularization constraints.
    """
    inputs = tf.keras.Input(shape=base_model.input_shape[1:])
    x = inputs
    if preprocessing_layer is not None:
        x = preprocessing_layer(x)
    
    x = base_model(x)
    gap = GlobalAveragePooling2D()(x)
    gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Concatenate()([gap, gmp])
    x = BatchNormalization()(x)
    x = Dense(
        dense_units_1,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = Dropout(dropout_rate_1)(x)
    x = BatchNormalization()(x)
    x = Dense(
        dense_units_2,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
    )(x)
    x = Dropout(dropout_rate_2)(x)
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model


def build_mobilenetv2_model(
    num_classes: int, 
    img_size: int = 224,
    dense_units_1: int = 512,
    dense_units_2: int = 256,
    dropout_rate_1: float = 0.5,
    dropout_rate_2: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001
) -> Tuple[Model, Model]:
    """
    Builds a MobileNetV2 model for classification with configurable layer constraints.
    """
    try:
        logger.info(f"Building MobileNetV2 model for {num_classes} classes...")
        input_shape = (img_size, img_size, 3)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze base model
        base_model.trainable = False
        
        # MobileNetV2 expects input in [-1, 1]. Since input is normalized to [0, 1],
        # we scale by 2.0 and shift by -1.0.
        preprocessing_layer = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)
        
        model = _build_custom_head(
            base_model=base_model,
            num_classes=num_classes,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dropout_rate_1=dropout_rate_1,
            dropout_rate_2=dropout_rate_2,
            l2_reg=l2_reg,
            preprocessing_layer=preprocessing_layer
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("MobileNetV2 model successfully built and compiled.")
        return model, base_model
        
    except Exception as e:
        logger.error(f"Error building MobileNetV2 model: {str(e)}")
        raise

def build_efficientnetb0_model(
    num_classes: int, 
    img_size: int = 224,
    dense_units_1: int = 512,
    dense_units_2: int = 256,
    dropout_rate_1: float = 0.5,
    dropout_rate_2: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001
) -> Tuple[Model, Model]:
    """
    Builds an EfficientNetB0 model for classification with configurable layer constraints.
    """
    try:
        logger.info(f"Building EfficientNetB0 model for {num_classes} classes...")
        input_shape = (img_size, img_size, 3)
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze base model
        base_model.trainable = False
        
        # EfficientNet expects input in [0, 255] because it has internal rescaling.
        # Since our generators rescale by 1./255, we rescale by 255.0 to restore it.
        preprocessing_layer = tf.keras.layers.Rescaling(scale=255.0)
        
        model = _build_custom_head(
            base_model=base_model,
            num_classes=num_classes,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dropout_rate_1=dropout_rate_1,
            dropout_rate_2=dropout_rate_2,
            l2_reg=l2_reg,
            preprocessing_layer=preprocessing_layer
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("EfficientNetB0 model successfully built and compiled.")
        return model, base_model
        
    except Exception as e:
        logger.error(f"Error building EfficientNetB0 model: {str(e)}")
        raise

def build_efficientnetb3_model(
    num_classes: int, 
    img_size: int = 224,
    dense_units_1: int = 512,
    dense_units_2: int = 256,
    dropout_rate_1: float = 0.5,
    dropout_rate_2: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001
) -> Tuple[Model, Model]:
    """
    Builds an EfficientNetB3 model for classification with configurable layer constraints.
    """
    try:
        logger.info(f"Building EfficientNetB3 model for {num_classes} classes...")
        input_shape = (img_size, img_size, 3)
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze base model
        base_model.trainable = False
        
        # EfficientNet expects input in [0, 255] because it has internal rescaling.
        # Since our generators rescale by 1./255, we rescale by 255.0 to restore it.
        preprocessing_layer = tf.keras.layers.Rescaling(scale=255.0)
        
        model = _build_custom_head(
            base_model=base_model,
            num_classes=num_classes,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dropout_rate_1=dropout_rate_1,
            dropout_rate_2=dropout_rate_2,
            l2_reg=l2_reg,
            preprocessing_layer=preprocessing_layer
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("EfficientNetB3 model successfully built and compiled.")
        return model, base_model
        
    except Exception as e:
        logger.error(f"Error building EfficientNetB3 model: {str(e)}")
        raise

def build_resnet50_model(
    num_classes: int, 
    img_size: int = 224,
    dense_units_1: int = 512,
    dense_units_2: int = 256,
    dropout_rate_1: float = 0.5,
    dropout_rate_2: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001
) -> Tuple[Model, Model]:
    """
    Builds a ResNet50 model for classification with configurable layer constraints.
    """
    try:
        logger.info(f"Building ResNet50 model for {num_classes} classes...")
        input_shape = (img_size, img_size, 3)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze base model
        base_model.trainable = False
        
        # ResNet50 expects BGR with ImageNet mean subtraction.
        # We rescale back to [0, 255] and apply ResNet50's preprocess_input.
        preprocessing_layer = tf.keras.Sequential([
            tf.keras.layers.Rescaling(scale=255.0),
            tf.keras.layers.Lambda(lambda img: tf.keras.applications.resnet50.preprocess_input(img))
        ])
        
        model = _build_custom_head(
            base_model=base_model,
            num_classes=num_classes,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dropout_rate_1=dropout_rate_1,
            dropout_rate_2=dropout_rate_2,
            l2_reg=l2_reg,
            preprocessing_layer=preprocessing_layer
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("ResNet50 model successfully built and compiled.")
        return model, base_model
        
    except Exception as e:
        logger.error(f"Error building ResNet50 model: {str(e)}")
        raise

def unfreeze_top_layers(model: Model, base_model: Model, num_layers: int = 30, learning_rate: float = 0.0001) -> Model:
    """
    Unfreezes the top N layers of the base model for fine-tuning.
    
    Args:
        model (Model): The complete compiled model.
        base_model (Model): The base model architecture inside the complete model.
        num_layers (int): Number of layers from the top of the base model to unfreeze.
        learning_rate (float): Recompilation learning rate for fine-tuning.
        
    Returns:
        Model: Recompiled model ready for fine-tuning.
    """
    try:
        logger.info(f"Unfreezing the top {num_layers} layers of the base model...")
        
        # Ensure base model is trainable
        base_model.trainable = True
        
        # Freeze all layers except the last 'num_layers'
        total_layers = len(base_model.layers)
        for layer in base_model.layers[:total_layers - num_layers]:
            layer.trainable = False
            
        for layer in base_model.layers[total_layers - num_layers:]:
            # Keep BatchNormalization layers frozen to prevent weight destruction
            # Common best practice for fine-tuning
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                
        # Count trainable layers in the full model
        trainable_count = sum([1 for layer in model.layers if layer.trainable])
        logger.info(f"Number of trainable layers in the entire model: {trainable_count}")
        
        # Recompile with lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model recompiled for fine-tuning.")
        return model
        
    except Exception as e:
        logger.error(f"Error unfreezing top layers: {str(e)}")
        raise

def get_model_summary_info(model: Model) -> Dict[str, Any]:
    """
    Retrieves summary information about the model's parameters and estimated size.
    
    Args:
        model (Model): The Keras model.
        
    Returns:
        Dict[str, Any]: Dictionary containing total_params, trainable_params, 
                        non_trainable_params, and model_size_mb.
    """
    try:
        logger.info("Calculating model summary information...")
        
        trainable_count = int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
        non_trainable_count = int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
        total_params = trainable_count + non_trainable_count
        
        # Calculate size assuming float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 ** 2)
        
        info = {
            "total_params": total_params,
            "trainable_params": trainable_count,
            "non_trainable_params": non_trainable_count,
            "model_size_mb": round(size_mb, 2)
        }
        
        logger.info(f"Summary info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Error getting model summary info: {str(e)}")
        raise

def save_model_all_formats(model: Model, save_dir: str) -> None:
    """
    Saves the model in multiple formats: .h5, SavedModel (pb), and TFLite.
    
    Args:
        model (Model): The Keras model to save.
        save_dir (str): Directory where the models should be saved.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created save directory: {save_dir}")
            
        logger.info(f"Saving models to {save_dir}...")
        
        # 1. Save as .h5
        h5_path = os.path.join(save_dir, 'model.h5')
        model.save(h5_path)
        h5_size = os.path.getsize(h5_path) / (1024 ** 2)
        logger.info(f"Saved .h5 format: {h5_path} ({h5_size:.2f} MB)")
        
        # 2. Save as SavedModel
        saved_model_path = os.path.join(save_dir, 'saved_model')
        if hasattr(model, 'export'):
            model.export(saved_model_path)
        else:
            model.save(saved_model_path)
        
        # Calculate SavedModel total size
        sm_size = 0
        for dirpath, _, filenames in os.walk(saved_model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                sm_size += os.path.getsize(fp)
        sm_size = sm_size / (1024 ** 2)
        logger.info(f"Saved SavedModel format: {saved_model_path} ({sm_size:.2f} MB)")
        
        # 3. Convert and save as TFLite
        try:
            logger.info("Converting to TFLite format with DEFAULT optimization...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = os.path.join(save_dir, 'model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            tflite_size = os.path.getsize(tflite_path) / (1024 ** 2)
            logger.info(f"Saved TFLite format: {tflite_path} ({tflite_size:.2f} MB)")
        except Exception as tflite_err:
            logger.warning(f"Could not convert model to TFLite format: {str(tflite_err)}")
        
        logger.info("All model formats saved successfully.")
        
    except Exception as e:
        logger.error(f"Error saving model formats: {str(e)}")
        raise
