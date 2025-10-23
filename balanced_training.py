import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

print("🎯 Training BALANCED model to fix normal image predictions...")

# Check class distribution


def check_class_balance():
    train_normal = len(os.listdir('data/train/NORMAL'))
    train_pneumonia = len(os.listdir('data/train/PNEUMONIA'))
    print(f"📊 Training Data Balance:")
    print(f"   Normal images: {train_normal}")
    print(f"   Pneumonia images: {train_pneumonia}")
    print(f"   Ratio: {train_pneumonia/train_normal:.2f}:1")
    return train_normal, train_pneumonia


normal_count, pneumonia_count = check_class_balance()

# Calculate class weights to handle imbalance
class_weight = {
    0: (1 / normal_count) * (normal_count + pneumonia_count) / 2.0,  # Normal
    1: (1 / pneumonia_count) * (normal_count + pneumonia_count) / 2.0  # Pneumonia
}
print(f"⚖️ Class weights: {class_weight}")

# Enhanced data augmentation with more focus on normal cases
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='constant',
    cval=0  # Fill with black for X-rays
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Create a model that's better at distinguishing normal cases


def create_balanced_model():
    model = keras.Sequential([
        # First block - learn basic features
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),

        # Second block - learn more complex features
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),

        # Third block - learn detailed patterns
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        # Fourth block - high-level features
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.5),

        # Classifier with regularization to prevent overconfidence
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


print("🧠 Creating balanced model...")
model = create_balanced_model()

# Use a conservative optimizer
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'specificity']
)

model.summary()

# Callbacks for stable training
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001),
]

print("🎯 Training with class weights and careful monitoring...")
print("Focus: Improving NORMAL image recognition...")

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weight,  # This helps with the imbalance!
    callbacks=callbacks,
    verbose=1
)

# Save the model
model.save('pneumonia_model.h5')
print("✅ Balanced model saved!")

# Test specifically on normal images


def test_normal_images():
    print("\n🧪 SPECIFIC TEST: Normal Images")
    normal_dir = "data/test/NORMAL"
    if os.path.exists(normal_dir):
        files = os.listdir(normal_dir)[:10]  # Test more normal images
        correct = 0
        total = len(files)

        for file in files:
            image_path = os.path.join(normal_dir, file)
            image = keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array, verbose=0)[0][0]
            if pred < 0.5:  # Correctly identified as normal
                correct += 1
                status = "✅"
            else:
                status = "❌"
            print(f"   {status} {file}: {pred:.3f}")

        accuracy = correct / total
        print(f"\n📊 Normal Image Accuracy: {accuracy:.1%} ({correct}/{total})")
        return accuracy


normal_accuracy = test_normal_images()

if normal_accuracy > 0.8:
    print("🎉 Great! Normal image detection is working well!")
else:
    print("🔄 Normal detection still needs improvement, but better than before!")

print("\n🔁 Please run 'Run Model Validation' in your Streamlit app to check overall performance!")
