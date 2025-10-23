import os

print("📁 Current directory:", os.getcwd())
print("\n📋 Files in current directory:")
for file in os.listdir('.'):
    if os.path.isfile(file):
        print(f"  - {file}")

print("\n🔍 Checking for required files:")
required_files = ['app.py', 'pneumonia_model.h5', 'grad_cam.py']
for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file} - FOUND")
    else:
        print(f"  ❌ {file} - MISSING")

print("\n📊 Model file size (if exists):")
if os.path.exists('pneumonia_model.h5'):
    size = os.path.getsize('pneumonia_model.h5') / (1024 * 1024)  # Convert to MB
    print(f"  pneumonia_model.h5: {size:.2f} MB")