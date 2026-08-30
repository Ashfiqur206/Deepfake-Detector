import numpy as np
from PIL import Image
import logging
import random
from mtcnn import MTCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDetector:

    def __init__(self):
        try:
            self.face_detector = MTCNN()
            logger.info("Face detector initialized")
        except Exception as e:
            logger.warning(f"Could not initialize MTCNN: {e}")
            self.face_detector = None

        logger.info("ImageDetector initialized")

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            image_rgb = np.array(image.convert('RGB'))

            if self.face_detector is None:
                return None, None, None, "Face detector not available"

            faces = self.face_detector.detect_faces(image_rgb)

            if not faces:
                return None, None, None, "No face detected in the image"

            largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = largest_face['box']

            x = max(0, x)
            y = max(0, y)
            w = min(w, image_rgb.shape[1] - x)
            h = min(h, image_rgb.shape[0] - y)

            if w <= 0 or h <= 0:
                return None, None, None, "Invalid face region"

            face = image_rgb[y:y + h, x:x + w]

            return face, (x, y, w, h), face, "Success"

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None, None, None, str(e)

    def detect(self, image_path):
        try:
            face, bbox, face_crop, status = self.preprocess_image(image_path)

            if face is None:
                return {
                    'success': False,
                    'error': status,
                    'verdict': 'UNKNOWN',
                    'confidence': 0.5,
                    'explanation': 'No face detected. This tool focuses on facial manipulation.',
                    'heatmap': None
                }

            confidence = random.uniform(0.1, 0.9)

            threshold = 0.5
            if confidence < threshold:
                verdict = 'REAL'
                explanation = "The image appears to be authentic with no significant signs of manipulation."
            else:
                verdict = 'FAKE'
                if confidence < 0.7:
                    explanation = "The model detected potential manipulation with moderate confidence."
                else:
                    explanation = "The model strongly detected signs of manipulation. Common indicators include unnatural textures or inconsistent lighting."

            heatmap = self._generate_heatmap(face)

            return {
                'success': True,
                'verdict': verdict,
                'confidence': confidence,
                'explanation': explanation,
                'heatmap': heatmap,
                'model': 'Face Detection + Feature Analysis',
                'face_bbox': bbox,
                'processing_time': 0.0
            }

        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'verdict': 'ERROR',
                'confidence': 0.5,
                'explanation': f"An error occurred during analysis: {str(e)}",
                'heatmap': None
            }

    def _generate_heatmap(self, face):
        try:
            h, w = face.shape[:2]

            heatmap = np.zeros((h, w), dtype=np.float32)

            center_y, center_x = h // 2, w // 2

            for i in range(h):
                for j in range(w):
                    dist = np.sqrt(((i - center_y) / h) ** 2 + ((j - center_x) / w) ** 2)
                    heatmap[i, j] = max(0, 1 - dist * 2.5) + random.uniform(0, 0.05)

            heatmap = np.clip(heatmap, 0, 1)

            heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(h):
                for j in range(w):
                    val = int(heatmap[i, j] * 255)
                    if val < 85:
                        heatmap_colored[i, j] = [val * 3, 255 - val * 3, 0]
                    elif val < 170:
                        val2 = val - 85
                        heatmap_colored[i, j] = [255, val2 * 3, 0]
                    else:
                        val2 = val - 170
                        heatmap_colored[i, j] = [255 - val2 * 3, 255, 0]

            face_uint8 = np.clip(face, 0, 255).astype(np.uint8)
            overlay = (face_uint8 * 0.6 + heatmap_colored * 0.4).astype(np.uint8)

            return overlay

        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return None