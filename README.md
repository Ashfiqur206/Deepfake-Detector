# 🛡️ DeepFake Sentinel

DeepFake Sentinel is an AI-powered tool that helps people detect synthetic media (deepfakes) in images, videos, and audio files.

## 🚀 Features

- **Multi-Modal Detection**: Analyze images, videos, and audio in one place
- **User-Friendly Interface**: Simple drag-and-drop upload with clear results
- **Explainable AI**: See why the model made its decision with heatmaps and explanations
- **Privacy-First**: Files are processed in memory and immediately discarded
- **Educational**: Learn about deepfakes while using the tool

## 📦 Installation

### Prerequisites

- Python 3.12 
- FFmpeg (for video processing)
- 4GB+ RAM recommended

### Quick Start

```bash
git clone https://github.com/Ashfiqur206/Deepfake-Detector.git
cd deepfake-sentinel
python -m venv venv1
source venv1/bin/activate
pip install -r requirements.txt
streamlit run app.py