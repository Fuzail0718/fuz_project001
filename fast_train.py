import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
print("🚀 Starting training script...")


print("✅ TensorFlow imported successfully")

# Check if dataset exists
print("📁 Checking for dataset...")
if not os.path.exists('data/train'):
    print("❌ ERROR: Dataset folder 'data/train' not found!")
    print("💡 Make sure you have the data folder with train/test/val subfolders")
    exit()
else:
    print("✅ Dataset folder found!")

# Check what's in the data folder
print("📊 Contents of data folder:")
for item in os.listdir('data'):
    print(f"  - {item}")

print("🧠 Creating model...")

# Simple model for quick training
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("✅ Model created!")

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

print("📊 Loading training data...")

# Data generator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

print(f"✅ Loaded {train_generator.samples} training images")
print("🎯 Starting training (2 epochs)...")

# Quick training
model.fit(train_generator, epochs=2, steps_per_epoch=10, verbose=1)

print("💾 Saving model...")
model.save('pneumonia_model.h5')
print("🎉 pneumonia_model.h5 created successfully!")
print("📏 Model file size:", os.path.getsize('pneumonia_model.h5'), "bytes")
