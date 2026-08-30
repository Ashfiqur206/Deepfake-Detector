import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from models.image_detector import ImageDetector
    print("✅ ImageDetector imported successfully")
except Exception as e:
    print(f"❌ Failed to import ImageDetector: {e}")

try:
    from models.video_detector import VideoDetector
    print("✅ VideoDetector imported successfully")
except Exception as e:
    print(f"❌ Failed to import VideoDetector: {e}")

try:
    from models.audio_detector import AudioDetector
    print("✅ AudioDetector imported successfully")
except Exception as e:
    print(f"❌ Failed to import AudioDetector: {e}")

print("\nCurrent Python path:")
for path in sys.path:
    print(f"  {path}")