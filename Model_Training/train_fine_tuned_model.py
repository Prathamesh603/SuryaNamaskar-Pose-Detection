import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Define Dataset Path (Make Sure Your Dataset is Structured Correctly)
dataset_path = "D:\\ET23BTCO123\\suraya namskar pose detection\\data for CNN\\dataset"  # Ensure 'dataset/' contains 8 pose folders with images

#2. Load and Augment Data
data_gen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% data for validation
)

#3. Create Training and Validation Sets
train_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # MobileNetV2 input size
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Get Pose Class Names
class_labels = list(train_data.class_indices.keys())
print("Pose Classes:", class_labels)

#  Load Pretrained MobileNetV2 Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers to use as feature extractor

#Build the Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)  # Prevent overfitting
output = Dense(len(class_labels), activation="softmax")(x)  # Output layer with pose labels

model = Model(inputs=base_model.input, outputs=output)

#Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

#Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # Adjust based on performance
    verbose=1
)

#Save the Model
model.save("fine_tuned_mobilenetv2.h5")
print("Model Training Complete & Saved!")

# Plot Accuracy & Loss Graphs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.show()
