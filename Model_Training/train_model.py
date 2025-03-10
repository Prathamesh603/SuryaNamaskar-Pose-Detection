import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# -------------------------------
# Define Constants
# -------------------------------
IMG_SIZE = (224, 224)   # Input image size
BATCH_SIZE = 32         # Number of images per batch
EPOCHS = 15             # Number of training epochs
NUM_CLASSES = 8         # Number of Surya Namaskar poses
DATASET_PATH = r"D:\ET23BTCO123\suraya namskar pose detection\data for CNN\dataset"  # Update this path

# -------------------------------
# Data Preprocessing & Augmentation
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,        # Normalize pixel values
    rotation_range=20,      # Random rotation
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1, # Vertical shift
    shear_range=0.1,        # Shear transformation
    zoom_range=0.1,         # Zoom-in and zoom-out
    horizontal_flip=True,   # Flip images horizontally
    validation_split=0.2    # 80% training, 20% validation
)

# Load Training Data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load Validation Data
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -------------------------------
# Load Pretrained MobileNetV2 Model
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers to retain pretrained weights
base_model.trainable = False

# -------------------------------
# Build the Model
# -------------------------------
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Reduce overfitting
    Dense(NUM_CLASSES, activation='softmax')  # Output layer with 8 classes
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary
model.summary()

# -------------------------------
# Train the Model
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -------------------------------
# Save the Model
# -------------------------------
model.save("surya_namaskar_pose_model.h5")
print("✅ Model saved successfully!")

# -------------------------------
# Evaluate the Model
# -------------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.show()

# Plot the training history
plot_training_history(history)

# -------------------------------
# Test the Model on a Single Image
# -------------------------------
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_pose(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get class labels
    class_labels = list(train_generator.class_indices.keys())
    print(f"📌 Predicted Pose: {class_labels[predicted_class]}")

# Example Test (Update with an actual image path)
test_image_path = "D:\ET23BTCO123\suraya namskar pose detection\test\STEP-BY-STEP-SURYA-NAMASKAR-FOR-BEGINNERS-_-Learn-Sun-Salutation-In-3-Minutes_-Simple-Yoga-Lessons-YouTube-Google-Chrome-2024-06-18-12-08-40_mp4-0013_jpg.rf.7807110393d9d156f842c29522a3ab71.jpg"  # Update this path
predict_pose(test_image_path)
