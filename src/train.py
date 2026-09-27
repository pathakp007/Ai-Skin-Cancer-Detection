import os
import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB0
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ======================
# Paths
# ======================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "dataset", "processed_dataset", "train")
valid_dir = os.path.join(BASE_DIR, "dataset", "processed_dataset", "valid")
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)

# ======================
# Parameters
# ======================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# Load Dataset
# ======================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

# ======================
# Data Augmentation
# ======================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# ======================
# Base Model
# ======================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ======================
# Build Model
# ======================

inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# Callbacks
# ======================

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "skin_cancer_model.keras"),
    save_best_only=True
)

early_stop = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

# ======================
# Train
# ======================

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

# ======================
# Save Model
# ======================

model.save(os.path.join(model_dir, "skin_cancer_model.keras"))

print("Training Complete!")