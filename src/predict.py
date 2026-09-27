import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ==============================
# Project Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "skin_cancer_model.keras"
)

# ==============================
# Load Trained Model
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# Disease Classes
# (Must match your training folder names)
# ==============================

classes = [
    "Actinic_Keratoses",
    "Basal_Cell_Carcinoma",
    "Benign_Keratosis",
    "Dermatofibroma",
    "Melanocytic_Nevi",
    "Melanoma",
    "Vascular_Lesion"
]

# ==============================
# Prediction Function
# ==============================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224,224))

    image = np.array(image)

    image = image.astype("float32")

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    class_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    disease = classes[class_index]

    return disease, confidence