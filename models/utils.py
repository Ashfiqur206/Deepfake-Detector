import numpy as np
import cv2
from PIL import Image
import io
import base64
import os


def format_confidence(confidence_score):
    return f"{confidence_score * 100:.1f}%"


def create_explanation(verdict, confidence, model_type):
    explanations = {
        'image': {
            'REAL': [
                "The image appears to be authentic with no significant signs of manipulation.",
                "The facial features are consistent with natural patterns.",
                "No obvious artifacts from AI generation were detected."
            ],
            'FAKE': [
                "The image shows signs of AI manipulation.",
                "The model detected inconsistencies in facial features.",
                "Look for unnatural blending around the edges or irregular textures."
            ]
        },
        'video': {
            'REAL': [
                "The video appears consistent across all analyzed frames.",
                "No significant signs of manipulation were detected."
            ],
            'FAKE': [
                "The video shows signs of manipulation in multiple frames.",
                "The model detected inconsistencies across the video sequence."
            ]
        },
        'audio': {
            'REAL': [
                "The audio shows natural human speech characteristics.",
                "Breath patterns and natural variations were detected."
            ],
            'FAKE': [
                "The audio shows signs of being AI-generated.",
                "The speech patterns are unnaturally smooth."
            ]
        }
    }

    model_explanations = explanations.get(model_type, explanations['image'])
    verdict_explanations = model_explanations.get(verdict, model_explanations['REAL'])

    if confidence > 0.8:
        return verdict_explanations[0] if verdict_explanations else "High confidence detection."
    elif confidence > 0.6:
        return verdict_explanations[1] if len(verdict_explanations) > 1 else "Moderate confidence detection."
    else:
        return verdict_explanations[2] if len(verdict_explanations) > 2 else "Low confidence detection."


def preprocess_audio(audio_path, target_sr=16000):
    import librosa

    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio, sr
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None


def extract_audio_features(audio, sr, n_mfcc=13):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)

        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)

        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)

        return {
            'mfccs_mean': mfccs_mean,
            'mfccs_std': mfccs_std,
            'spectral_centroid': spectral_centroid_mean,
            'zero_crossing_rate': zcr_mean,
            'rms_energy': rms_mean
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def create_audio_heatmap(audio, sr, feature_type='spectrogram'):
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    time = np.linspace(0, len(audio) / sr, len(audio))
    axes[0].plot(time, audio)
    axes[0].set_title('Audio Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


def validate_file_type(filename, allowed_types=None):
    if allowed_types is None:
        allowed_types = {
            'image': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            'video': ['mp4', 'avi', 'mov', 'mkv', 'webm'],
            'audio': ['mp3', 'wav', 'm4a', 'flac', 'ogg']
        }

    all_types = [ext for types in allowed_types.values() for ext in types]

    ext = filename.split('.')[-1].lower()

    if ext not in all_types:
        return None, f"Unsupported file type: {ext}. Supported types: {', '.join(all_types)}"

    for media_type, extensions in allowed_types.items():
        if ext in extensions:
            return media_type, None

    return None, "Unknown file type"


def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def create_result_visualization(result, media_type):
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    confidence = result.get('confidence', 0.5)
    colors = ['#d4edda', '#fff3cd', '#f8d7da']
    color_idx = 0 if confidence < 0.4 else 1 if confidence < 0.7 else 2

    ax1.barh(['Confidence'], [confidence], color=colors[color_idx])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Confidence Score')
    ax1.set_title(f"Verdict: {result.get('verdict', 'UNKNOWN')}")
    ax1.text(confidence / 2, 0, f"{confidence * 100:.1f}%",
             ha='center', va='center', fontsize=14, fontweight='bold')

    explanation = result.get('explanation', 'No explanation available')
    ax2.text(0.5, 0.5, explanation, ha='center', va='center',
             wrap=True, fontsize=10)
    ax2.set_title('Explanation')
    ax2.axis('off')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf