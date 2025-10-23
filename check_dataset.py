import os


def check_dataset_structure():
    # Try different possible folder names
    possible_paths = [
        'data/train',
        'chest_xray/train',
        'chest_xray/chest_xray/train'  # Sometimes it's nested
    ]

    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            print(f"✅ Found dataset at: {path}")
            break

    if not actual_path:
        print("❌ Could not find dataset folder. Current directory contents:")
        print(os.listdir('.'))
        return None

    # Now check the actual structure
    base_path = actual_path.split('/train')[0]  # Get the base folder

    required_folders = ['train', 'test', 'val']
    required_classes = ['PNEUMONIA', 'NORMAL']

    print(f"\n📁 Checking structure under: {base_path}/")

    for folder in required_folders:
        for class_name in required_classes:
            path = os.path.join(base_path, folder, class_name)
            if not os.path.exists(path):
                print(f"❌ Missing: {path}")
                return None
            num_files = len([f for f in os.listdir(
                path) if f.endswith(('.jpeg', '.jpg', '.png'))])
            print(f"✅ {path}: {num_files} images")

    print(f"\n🎉 Dataset structure is correct! Using base path: {base_path}")
    return base_path


# Run the check
base_path = check_dataset_structure()
if base_path:
    print(
        f"\nUpdate your training script to use: '{base_path}/train' and '{base_path}/val'")
else:
    print("\nPlease check your dataset organization.")
