from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.padel.settings import get_env


@dataclass
class PlayerDetection:
    frame_index: int
    player_id: Optional[int]
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str


class YOLOHandler:
    """
    YOLO wrapper to load weights and run inference for player detection.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        model_name: str = "yolov8n.pt",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = self._resolve_weights(weights_path, model_name)
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        model = YOLO(str(self.weights_path))
        model.to(self.device)
        return model

    def _resolve_weights(self, weights_path: Optional[Path], model_name: str) -> Path:
        if weights_path:
            return Path(weights_path)
        
        # Try to get from env, otherwise use default model name (ultralytics will download it)
        env_weights = get_env("YOLO_WEIGHTS", required=False)
        if env_weights:
            return Path(env_weights)
        
        return Path(model_name)

    def predict_frame(self, frame: np.ndarray, frame_index: int) -> List[PlayerDetection]:
        """
        Run inference on a single frame.
        """
        results = self.model.predict(frame, device=self.device, verbose=False)
        
        detections: List[PlayerDetection] = []
        
        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            # We only care about persons (class 0 in COCO)
            cls_idx = int(boxes.cls[i])
            if cls_idx != 0:
                continue
                
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i])
            
            # Get track ID if available
            track_id = int(boxes.id[i]) if boxes.id is not None else None
            
            detections.append(
                PlayerDetection(
                    frame_index=frame_index,
                    player_id=track_id,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=conf,
                    class_name="person"
                )
            )
            
        return detections

    def run_on_video(
        self, video_path: Path, start_frame: int = 0
    ) -> Iterable[PlayerDetection]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")

        frame_index = start_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.predict_frame(frame, frame_index)
            for det in detections:
                yield det
            
            frame_index += 1

        cap.release()
