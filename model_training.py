import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

print("🔍 Starting training script...")
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Current directory:", os.getcwd())

# Set parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

def verify_paths():
    """Verify all required paths exist"""
    base_path = 'data'
    
    if not os.path.exists(base_path):
        print(f"❌ Base path '{base_path}' not found!")
        print("Current directory contents:", os.listdir('.'))
        return False
    
    required_paths = [
        'train/PNEUMONIA',
        'train/NORMAL', 
        'test/PNEUMONIA',
        'test/NORMAL',
        'val/PNEUMONIA',
        'val/NORMAL'
    ]
    
    all_paths_exist = True
    for path in required_paths:
        full_path = os.path.join(base_path, path)
        if not os.path.exists(full_path):
            print(f"❌ Missing: {full_path}")
            all_paths_exist = False
        else:
            num_files = len([f for f in os.listdir(full_path) if f.endswith(('.jpeg', '.jpg', '.png'))])
            print(f"✅ {full_path}: {num_files} images")
    
    return all_paths_exist

def main():
    print("📊 Verifying dataset paths...")
    if not verify_paths():
        print("❌ Dataset structure incorrect. Please check folder organization.")
        return
    
    print("🚀 Setting up data generators...")
    try:
        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        # Load data - using absolute paths to be safe
        train_path = os.path.join('data', 'train')
        val_path = os.path.join('data', 'val')
        
        print(f"📁 Training path: {train_path}")
        print(f"📁 Validation path: {val_path}")

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"✅ Found {train_generator.samples} training images")
        print(f"✅ Found {val_generator.samples} validation images")
        print(f"✅ Class indices: {train_generator.class_indices}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🧠 Creating model...")
    try:
        # Use a simpler approach first
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model created and compiled successfully")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🚀 Starting training...")
    try:
        # Use smaller steps for testing
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            verbose=1,
            steps_per_epoch=min(50, train_generator.samples // BATCH_SIZE),
            validation_steps=min(20, val_generator.samples // BATCH_SIZE)
        )
        
        # Save the model
        model.save('pneumonia_model.h5')
        print("✅ Model saved as 'pneumonia_model.h5'")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.savefig('training_history.png')
        print("✅ Training history saved as 'training_history.png'")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()