import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Parameters
image_size = (64, 64)               # Resize images to this size
batch_size = 10                      # Number of images per batch
num_classes = 3                # Number of classes (A-Z)

# Paths
dataset_path = 'C:/Users/ROHIT/Downloads/SIGN/Gesture Image Data'  # Path to dataset directory
model_save_path = 'sign_language_model.h5'  # File path to save the model

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Split 20% of data for validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training and Validation Data Generators
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
epochs = 15
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the Model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
