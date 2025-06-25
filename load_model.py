import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
def load_cxr_model(model_path='C:\mini project\cxr-analysis-project\model\nilesh.h5'):
    model = load_model(model_path)
    print("Model loaded successfully")
    return model

# Preprocess the image before making predictions
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to 0-1
    return img_array
