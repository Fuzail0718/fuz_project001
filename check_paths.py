import os

print("📂 Current working directory:", os.getcwd())
print("📁 Contents of current directory:")
for item in os.listdir('.'):
    print("  -", item)

print("\n📊 Checking data folder:")
if os.path.exists('data'):
    print("✅ 'data' folder exists")
    print("📁 Contents of data folder:")
    for item in os.listdir('data'):
        print("  -", item)
else:
    print("❌ 'data' folder NOT found")

print("\n🔍 Checking data/train:")
if os.path.exists('data/train'):
    print("✅ 'data/train' exists")
    print("📁 Contents of data/train:")
    for item in os.listdir('data/train'):
        print("  -", item)
else:
    print("❌ 'data/train' NOT found")
