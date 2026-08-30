import os
import sys
import subprocess
import platform


def setup_environment():
    print("🚀 Setting up DeepFake Sentinel environment...")

    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️ Python 3.8+ is required. Please upgrade your Python version.")
        return False

    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    directories = ['models', 'static', 'tests', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg is installed")
    except:
        print("⚠️ FFmpeg not found. Please install FFmpeg for video processing:")
        if platform.system() == "Windows":
            print("   Download from: https://ffmpeg.org/download.html")
        elif platform.system() == "Darwin":
            print("   Run: brew install ffmpeg")
        else:
            print("   Run: sudo apt-get install ffmpeg")

    print("\n✨ Setup complete!")
    print("\nTo run the application:")
    print("1. Activate virtual environment")
    print("2. Run: streamlit run app.py")

    return True


if __name__ == "__main__":
    setup_environment()