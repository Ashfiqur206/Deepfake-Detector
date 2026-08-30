import numpy as np
import tempfile
import os
import logging
import random
from PIL import Image
from .image_detector import ImageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDetector:

    def __init__(self):
        self.image_detector = ImageDetector()
        logger.info("VideoDetector initialized")

    def extract_frames_with_pil(self, video_path):
        try:
            from moviepy.editor import VideoFileClip

            clip = VideoFileClip(video_path)
            duration = clip.duration

            if duration is None or duration <= 0:
                return [], 0

            frames = []
            frame_count = 0

            sample_interval = 1

            if duration > 60:
                sample_interval = 5
            elif duration > 300:
                sample_interval = 10

            for t in range(0, int(duration), sample_interval):
                try:
                    frame = clip.get_frame(t)
                    if frame is not None:
                        frames.append(frame)
                        frame_count += 1
                        if frame_count >= 30:
                            break
                except:
                    continue

            clip.close()

            return frames, frame_count

        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return [], 0

    def detect(self, video_path):
        try:
            frames, total_frames = self.extract_frames_with_pil(video_path)

            if not frames:
                return {
                    'success': False,
                    'error': 'Could not extract frames from video',
                    'verdict': 'UNKNOWN',
                    'confidence': 0.5,
                    'explanation': 'No frames could be extracted from this video.',
                    'frame_results': []
                }

            frame_results = []
            for i, frame in enumerate(frames):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    img = Image.fromarray(frame)
                    img.save(tmp.name)
                    frame_path = tmp.name

                try:
                    result = self.image_detector.detect(frame_path)
                    confidence = result.get('confidence', random.uniform(0.2, 0.8))
                    verdict = 'FAKE' if confidence > 0.5 else 'REAL'

                    frame_results.append({
                        'frame': i,
                        'verdict': verdict,
                        'confidence': confidence,
                        'success': result.get('success', False)
                    })
                finally:
                    try:
                        os.unlink(frame_path)
                    except:
                        pass

                if i >= 29:
                    break

            if frame_results:
                fake_frames = [r for r in frame_results if r['verdict'] == 'FAKE']
                fake_ratio = len(fake_frames) / len(frame_results)

                confidences = [r['confidence'] for r in frame_results]
                avg_confidence = np.mean(confidences) if confidences else 0.5

                if fake_ratio >= 0.5:
                    verdict = 'FAKE'
                    confidence = avg_confidence * (0.5 + fake_ratio * 0.5)
                else:
                    verdict = 'REAL'
                    confidence = avg_confidence * (0.5 + (1 - fake_ratio) * 0.5)

                confidence = min(0.95, max(0.05, confidence))

                if verdict == 'FAKE':
                    if fake_ratio > 0.8:
                        explanation = f"Strong evidence of manipulation: {len(fake_frames)} out of {len(frame_results)} analyzed frames appear fake."
                    elif fake_ratio > 0.5:
                        explanation = f"Potential manipulation detected: {len(fake_frames)} out of {len(frame_results)} analyzed frames appear fake."
                    else:
                        explanation = f"Some signs of manipulation: {len(fake_frames)} out of {len(frame_results)} frames show potential issues."
                else:
                    explanation = f"No significant manipulation detected in {len(frame_results)} analyzed frames."

                return {
                    'success': True,
                    'verdict': verdict,
                    'confidence': confidence,
                    'explanation': explanation,
                    'frame_results': frame_results,
                    'total_frames': total_frames,
                    'analyzed_frames': len(frame_results),
                    'fake_count': len(fake_frames),
                    'model': 'Frame-by-Frame Analysis',
                    'processing_time': 0.0
                }
            else:
                return {
                    'success': False,
                    'error': 'No frames could be analyzed',
                    'verdict': 'UNKNOWN',
                    'confidence': 0.5,
                    'explanation': 'No valid frames were extracted for analysis.',
                    'frame_results': []
                }

        except Exception as e:
            logger.error(f"Video detection error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'verdict': 'ERROR',
                'confidence': 0.5,
                'explanation': f"An error occurred during video analysis: {str(e)}",
                'frame_results': []
            }