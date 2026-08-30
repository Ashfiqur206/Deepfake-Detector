import os
import sys
import streamlit as st
import tempfile
import time
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import logging
import random
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from models.image_detector import ImageDetector
    from models.video_detector import VideoDetector
    from models.audio_detector import AudioDetector

    logger.info("Models imported successfully")
except ImportError as e:
    logger.error(f"Error importing models: {e}")
    st.error(f"Error importing models: {e}")
    st.stop()

st.set_page_config(
    page_title="DeepFake Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .real-box {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        .fake-box {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        .confidence-meter {
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
            transition: width 0.3s ease;
        }
        .explanation-box {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .warning-box {
            background: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #ffc107;
            margin: 1rem 0;
        }
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #f0f2ff;
        }
        .file-info {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .large-file-warning {
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .stImage {
            max-width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)


if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'upload_progress' not in st.session_state:
    st.session_state.upload_progress = 0


@st.cache_resource
def load_detectors():
    with st.spinner("🔍 Loading detection models..."):
        try:
            image_detector = ImageDetector()
            video_detector = VideoDetector()
            audio_detector = AudioDetector()
            st.session_state.model_loaded = True
            logger.info("All models loaded successfully")
            return image_detector, video_detector, audio_detector
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            st.error(f"Error loading models: {str(e)}")
            return None, None, None


def format_file_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def display_confidence_meter(confidence_score):
    percentage = confidence_score * 100
    color = "#28a745" if confidence_score < 0.4 else "#ffc107" if confidence_score < 0.7 else "#dc3545"

    st.markdown(f"""
    <div class="confidence-meter">
        <div class="confidence-fill" style="width:{percentage}%; background: {color};"></div>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>Real (0%)</span>
        <span><strong>{percentage:.1f}%</strong> Synthetic</span>
        <span>Fake (100%)</span>
    </div>
    """, unsafe_allow_html=True)


def display_results(result):
    verdict = result['verdict']
    confidence = result['confidence']
    explanation = result['explanation']

    if verdict == "REAL":
        box_class = "real-box"
        emoji = "✅"
        title = "Likely Real"
    elif verdict == "FAKE":
        box_class = "fake-box"
        emoji = "⚠️"
        title = "Likely Fake"
    else:
        box_class = ""
        emoji = "❓"
        title = "Uncertain"

    st.markdown(f"""
    <div class="result-box {box_class}">
        <h2>{emoji} Verdict: {title}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Confidence Score")
    display_confidence_meter(confidence)

    st.subheader("📖 What We Found")
    st.markdown(f"""
    <div class="explanation-box">
        <p>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Used:**")
            st.code(result.get('model', 'N/A'))
            st.markdown("**Processing Time:**")
            st.code(f"{result.get('processing_time', 0):.2f} seconds")
        with col2:
            st.markdown("**Confidence Score:**")
            st.code(f"{confidence:.4f}")
            st.markdown("**Threshold:**")
            st.code("0.50")

    if 0.35 < confidence < 0.65:
        st.warning("⚠️ Low Confidence Warning: The model is uncertain about this prediction.")


def display_image_result(result, uploaded_file):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔥 What the Model Focused On")
        if result.get('heatmap') is not None:
            st.image(result['heatmap'], use_container_width=True)
            st.caption("Red areas indicate what influenced the model's decision most")
        else:
            st.info("No face detected in this image. We can only analyze images with faces.")

    display_results(result)


def display_video_result(result, uploaded_file):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎥 Video Preview")
        st.video(uploaded_file)

    with col2:
        st.subheader("📊 Frame Analysis")

        frame_results = result.get('frame_results', [])
        if frame_results:
            labels = [f"Frame {i + 1}" for i in range(len(frame_results))]
            values = [r['confidence'] for r in frame_results]
            colors = ['#dc3545' if r['verdict'] == 'FAKE' else '#28a745' for r in frame_results]

            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v * 100:.1f}%" for v in values],
                    textposition='outside',
                )
            ])
            fig.update_layout(
                title="Frame-by-Frame Analysis",
                yaxis_title="Synthetic Score",
                yaxis_range=[0, 1],
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            fake_count = sum(1 for r in frame_results if r['verdict'] == 'FAKE')
            total_frames = len(frame_results)
            st.info(f"📊 {fake_count} out of {total_frames} analyzed frames appear manipulated")

    display_results(result)


def display_audio_result(result, uploaded_file):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔊 Audio Playback")
        st.audio(uploaded_file)

    with col2:
        st.subheader("📈 Audio Analysis")

        if result.get('waveform_info'):
            info = result['waveform_info']
            metrics = [
                ("Duration", f"{info.get('duration', 0):.2f}s"),
                ("Sample Rate", f"{info.get('sample_rate', 0)} Hz"),
                ("Channels", str(info.get('channels', 1))),
            ]
            for label, value in metrics:
                st.metric(label, value)

    display_results(result)


def process_large_file(uploaded_file, file_extension):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        file_size = uploaded_file.size
        max_size = 1 * 1024 * 1024 * 1024

        if file_size > max_size:
            st.error(f"File size exceeds 1GB limit. Your file is {format_file_size(file_size)}")
            return None

        status_text.text(f"Uploading {format_file_size(file_size)} file...")

        chunk_size = 10 * 1024 * 1024
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        uploaded_data = bytearray()

        for i in range(total_chunks):
            chunk = uploaded_file.read(chunk_size)
            uploaded_data.extend(chunk)
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            status_text.text(f"Uploading: {progress * 100:.1f}% complete")

        status_text.text("Processing uploaded file...")
        progress_bar.progress(1.0)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
        temp_file.write(uploaded_data)
        temp_file.close()

        status_text.text("File ready for analysis!")

        return temp_file.name

    except Exception as e:
        logger.error(f"Error processing large file: {e}")
        st.error(f"Error processing file: {str(e)}")
        return None


def main():
    load_css()

    st.markdown("""
    <div class="main-header">
        <h1>🛡️ DeepFake Sentinel</h1>
        <p style="font-size: 1.1rem; margin: 0;">Helping You Spot Synthetic Media with AI</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        DeepFake Sentinel is an educational tool that analyzes images, videos, and audio 
        to detect potential deepfake manipulation.
        """)

        st.header("⚙️ How It Works")
        st.markdown("""
        1. **Upload** a file (image, video, or audio)
        2. **AI Analysis** - Our models examine the content
        3. **Get Results** - Verdict, confidence score, and explanation
        4. **Learn** - See what the model detected
        """)

        st.header("📌 Models Used")
        st.markdown("""
        - **Images**: Face detection + feature analysis
        - **Videos**: Frame-by-frame analysis
        - **Audio**: Voice pattern analysis
        """)

        st.header("📦 File Limits")
        st.markdown("""
        - **Maximum Size:** 1 GB
        - **Supported Formats:** Images, Videos, Audio
        """)

        st.header("⚠️ Important")
        st.warning("This tool provides best-guess estimates. It is not 100% accurate.")

        st.header("🔒 Privacy")
        st.markdown("""
        ✅ We process files in memory only  
        ✅ Files are immediately discarded  
        ✅ No data is stored or shared
        """)

        if st.session_state.model_loaded:
            st.success("✅ Models loaded successfully")
        else:
            st.warning("⏳ Models not loaded yet")

    st.markdown("### 📂 Upload Media for Analysis")
    st.info("📌 Maximum file size: 1 GB")

    uploaded_file = st.file_uploader(
        "Choose a file...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'm4a'],
        help="Supported formats: Images (jpg, png), Videos (mp4, avi, mov), Audio (mp3, wav, m4a). Max size: 1GB"
    )

    if uploaded_file is not None:
        try:
            file_size = uploaded_file.size
            max_size = 1 * 1024 * 1024 * 1024

            if file_size > max_size:
                st.error(f"File exceeds 1GB limit. Current size: {format_file_size(file_size)}")
                st.stop()

            st.markdown(f"""
            <div class="file-info">
                <strong>📄 File:</strong> {uploaded_file.name}<br>
                <strong>📊 Size:</strong> {format_file_size(file_size)}
            </div>
            """, unsafe_allow_html=True)

            image_detector, video_detector, audio_detector = load_detectors()

            if image_detector is None:
                st.error("Failed to load detection models.")
                return

            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_size > 100 * 1024 * 1024:
                st.markdown("""
                <div class="large-file-warning">
                    <strong>⚠️ Large File Detected</strong><br>
                    Processing large files may take several minutes. Please be patient.
                </div>
                """, unsafe_allow_html=True)

            temp_path = process_large_file(uploaded_file, file_extension)

            if temp_path is None:
                st.error("Failed to process uploaded file.")
                st.stop()

            try:
                with st.spinner("🔍 Analyzing your file..."):
                    start_time = time.time()

                    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                        st.markdown("### 📷 Image Analysis")
                        result = image_detector.detect(temp_path)
                        result['processing_time'] = time.time() - start_time
                        if result['success']:
                            display_image_result(result, uploaded_file)
                        else:
                            st.error(f"Image analysis failed: {result.get('error', 'Unknown error')}")

                    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
                        st.markdown("### 🎥 Video Analysis")
                        with st.spinner("Extracting and analyzing video frames..."):
                            result = video_detector.detect(temp_path)
                            result['processing_time'] = time.time() - start_time
                        if result['success']:
                            display_video_result(result, uploaded_file)
                        else:
                            st.error(f"Video analysis failed: {result.get('error', 'Unknown error')}")

                    elif file_extension in ['mp3', 'wav', 'm4a']:
                        st.markdown("### 🔊 Audio Analysis")
                        result = audio_detector.detect(temp_path)
                        result['processing_time'] = time.time() - start_time
                        if result['success']:
                            display_audio_result(result, uploaded_file)
                        else:
                            st.error(f"Audio analysis failed: {result.get('error', 'Unknown error')}")

                    else:
                        st.error(f"Unsupported file type: {file_extension}")

                if 'result' in locals() and result.get('success'):
                    st.session_state.detection_history.append({
                        'filename': uploaded_file.name,
                        'timestamp': time.time(),
                        'verdict': result.get('verdict', 'UNKNOWN'),
                        'confidence': result.get('confidence', 0.5),
                        'size': format_file_size(file_size)
                    })

                    with st.sidebar.expander("📝 Detection History"):
                        for item in st.session_state.detection_history[-10:]:
                            emoji = "✅" if item['verdict'] == 'REAL' else "⚠️" if item['verdict'] == 'FAKE' else "❓"
                            st.write(f"{emoji} {item['filename']} - {item['verdict']} ({item['confidence']:.2f})")
                            st.caption(f"Size: {item.get('size', 'N/A')}")

            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="upload-area">
                <h2>🖼️ Images</h2>
                <p>Upload JPG or PNG<br>with faces for analysis</p>
                <p style="font-size: 0.8rem; color: #6c757d;">Max: 1GB</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="upload-area">
                <h2>🎬 Videos</h2>
                <p>Upload MP4, AVI, or MOV<br>for frame-by-frame analysis</p>
                <p style="font-size: 0.8rem; color: #6c757d;">Max: 1GB</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="upload-area">
                <h2>🎵 Audio</h2>
                <p>Upload MP3, WAV, or M4A<br>for voice analysis</p>
                <p style="font-size: 0.8rem; color: #6c757d;">Max: 1GB</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        ### 💡 Tips for Large Files

        - **Videos**: Large videos will take longer to process. The system analyzes frames at intervals for efficiency.
        - **Images**: High-resolution images are resized for analysis while maintaining quality.
        - **Audio**: Long audio files are analyzed for patterns and characteristics.
        - **Performance**: Processing time depends on file size and complexity.
        """)


if __name__ == "__main__":
    main()