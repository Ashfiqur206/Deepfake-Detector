import numpy as np
from PIL import Image
import logging
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

logger = logging.getLogger(__name__)


class ImageDetector:
    FAKE_THRESHOLD = 0.60
    REAL_THRESHOLD = 0.40
    MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = SiglipForImageClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device).eval()

        cfg = self.model.config
        self.fake_index = 0
        for idx, label in (cfg.id2label or {}).items():
            if str(label).lower() in ("fake", "deepfake", "synthetic", "ai", "manipulated"):
                self.fake_index = int(idx)
                break

    def detect(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            fake_prob = self._extract_fake_prob(logits)
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
                        "name": "SigLIP Deepfake Detector",
                        "value": fake_prob,
                        "weight": 1.0,
                        "note": f"fake prob = {fake_prob:.3f}",
                    }
                ],
                "heatmap": None,
                "model": self.MODEL_NAME,
                "ensemble_method": "single-model",
                "detectors_voted": 1,
                "threshold": f"{self.REAL_THRESHOLD:.2f} / {self.FAKE_THRESHOLD:.2f}",
            }
        except Exception as e:
            logger.error(f"Image detect error: {e}", exc_info=True)
            return self._error_result(str(e))

    def _extract_fake_prob(self, logits):
        if hasattr(logits, "logits"):
            logits = logits.logits
        if isinstance(logits, (list, tuple)):
            logits = torch.tensor(logits)
        if not isinstance(logits, torch.Tensor):
            return 0.5

        flat = logits.detach().flatten()

        if flat.numel() == 0:
            return 0.5
        if flat.numel() == 1:
            return float(torch.sigmoid(flat[0]).item())

        probs = torch.softmax(flat, dim=0)
        idx = self.fake_index
        if idx >= probs.numel() or idx < 0:
            idx = 0

        return float(probs[idx].item())

    def _calibrate(self, raw_prob):
        confidence = float(np.clip(raw_prob, 0.02, 0.98))
        return confidence, self._band(confidence)

    @staticmethod
    def _band(c):
        if c < 0.20:
            return "very_low"
        if c < 0.40:
            return "low"
        if c < 0.60:
            return "medium"
        if c < 0.80:
            return "high"
        return "very_high"

    def _verdict(self, c):
        if c >= self.FAKE_THRESHOLD:
            return "FAKE"
        if c <= self.REAL_THRESHOLD:
            return "REAL"
        return "UNKNOWN"

    def _explain(self, verdict, confidence, raw):
        if verdict == "FAKE":
            return (
                f"The SigLIP model flagged this image as likely synthetic "
                f"(fake probability {raw:.0%}). Reported accuracy: ~94%."
            )
        if verdict == "REAL":
            return (
                f"No strong evidence of manipulation found "
                f"(fake probability {raw:.0%})."
            )
        return (
            f"The model is uncertain (fake probability {raw:.0%}). "
            f"Try a clearer image for a more decisive result."
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