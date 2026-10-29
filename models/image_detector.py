import numpy as np
from PIL import Image
import logging
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)


class ImageDetector:
    FAKE_THRESHOLD = 0.60
    REAL_THRESHOLD = 0.40
    MODEL_NAME = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = ViTForImageClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device).eval()
        # id2label: {"0": "real", "1": "fake"}
        self.fake_index = self._find_fake_index()

    def _find_fake_index(self):
        id2label = self.model.config.id2label
        for idx, label in id2label.items():
            if label.lower() == "fake":
                return int(idx)
        return 1  # fallback

    def detect(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            fake_prob = float(probs[self.fake_index])
            confidence, band = self._calibrate(fake_prob)
            verdict = self._verdict(confidence)

            return {
                "success": True,
                "verdict": verdict,
                "confidence": confidence,
                "confidence_band": band,
                "explanation": self._explain(verdict, confidence, fake_prob),
                "signals": [
                    {
                        "name": "CommunityForensics ViT",
                        "value": fake_prob,
                        "weight": 1.0,
                        "note": f"HF model output (fake prob = {fake_prob:.3f})",
                    }
                ],
                "heatmap": None,  # HF ViT does not provide Grad-CAM out of the box
                "model": "CommunityForensics-DeepfakeDet-ViT",
                "ensemble_method": "single-model",
                "detectors_voted": 1,
                "threshold": f"{self.REAL_THRESHOLD:.2f} / {self.FAKE_THRESHOLD:.2f}",
            }
        except Exception as e:
            logger.error(f"Image detect error: {e}", exc_info=True)
            return self._error_result(str(e))

    def _calibrate(self, raw_prob):
        # The model outputs a raw probability. Map to a calibrated confidence
        # that preserves the UNKNOWN band for borderline cases.
        # We keep raw_prob as-is but clip slightly to avoid 0/1 extremes.
        confidence = float(np.clip(raw_prob, 0.02, 0.98))
        return confidence, self._band(confidence)

    @staticmethod
    def _band(c):
        if c < 0.20: return "very_low"
        if c < 0.40: return "low"
        if c < 0.60: return "medium"
        if c < 0.80: return "high"
        return "very_high"

    def _verdict(self, c):
        if c >= self.FAKE_THRESHOLD: return "FAKE"
        if c <= self.REAL_THRESHOLD: return "REAL"
        return "UNKNOWN"

    def _explain(self, verdict, confidence, raw):
        if verdict == "FAKE":
            return (
                f"The CommunityForensics ViT flagged this image as likely synthetic "
                f"(raw fake probability {raw:.0%}). This model was trained on 2.7M images "
                f"across 4,803 generators and has a reported accuracy of 97.2%."
            )
        if verdict == "REAL":
            return (
                f"The model found no strong evidence of manipulation "
                f"(raw fake probability {raw:.0%}). The image is consistent with authentic content."
            )
        return (
            f"The model is uncertain (raw fake probability {raw:.0%}). "
            f"This often happens with heavy compression, unusual lighting, or images "
            f"from generators not seen during training. Try a clearer image."
        )

    def _error_result(self, msg):
        return {
            "success": False,
            "error": msg,
            "verdict": "ERROR",
            "confidence": 0.5,
            "explanation": f"Analysis failed: {msg}",
            "signals": [],
        }