"""Real-world Computer Vision Feature Tracking Module for BehaviorBench.

Uses OpenCV to track facial presence, head kinetic pose (pitch/yaw variance),
mouth movement dynamics, and illumination metrics from real MP4 video streams.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class VideoVisionTracker:
    """Extracts frame-by-frame visual features from video files using OpenCV."""

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        self.cascade_path = cascade_path
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from {cascade_path}")

    def track_video(
        self,
        video_path: str,
        target_fps: int = 30,
        max_duration_sec: Optional[float] = None,
    ) -> Dict:
        """Analyze video frames and extract dynamic facial and kinetic metrics.

        Args:
            video_path: Path to the .mp4 or supported video file.
            target_fps: Sampling framerate for normalized temporal analysis.
            max_duration_sec: Optional cap on video length to process.

        Returns:
            Dictionary containing summary statistics, mouth series, and frame-level records.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        native_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = native_total_frames / native_fps if native_fps > 0 else 0.0

        if max_duration_sec and duration_sec > max_duration_sec:
            duration_sec = max_duration_sec

        frame_step = max(1, int(round(native_fps / target_fps)))
        effective_fps = native_fps / frame_step

        frames_metadata: List[Dict] = []
        mouth_series: List[float] = []
        head_pitch_series: List[float] = []
        head_yaw_series: List[float] = []
        luminances: List[float] = []
        confidences: List[float] = []

        prev_mouth_roi: Optional[np.ndarray] = None
        last_known_face: Optional[Tuple[int, int, int, int]] = None
        face_lost_counter = 0

        frame_idx = 0
        processed_idx = 0

        # Anchor coordinates for neutral center
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        h_frame, w_frame = (first_frame.shape[0], first_frame.shape[1]) if ret else (240, 320)
        center_x_anchor = w_frame / 2.0
        center_y_anchor = h_frame / 2.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_duration_sec and (frame_idx / native_fps) > max_duration_sec:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp_sec = round(processed_idx / effective_fps, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lum = float(np.mean(gray))
            luminances.append(lum)

            # Detect face
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=3, minSize=(25, 25)
            )

            if len(faces) > 0:
                # Pick largest detected face
                best_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = best_face
                last_known_face = (x, y, w, h)
                face_lost_counter = 0
                face_detected = True
                conf = 0.95
            elif last_known_face is not None and face_lost_counter < 5:
                # Transient occlusion / tracking hold
                x, y, w, h = last_known_face
                face_lost_counter += 1
                face_detected = True
                conf = max(0.50, 0.95 - (face_lost_counter * 0.10))
            else:
                face_detected = False
                conf = 0.10
                x, y, w, h = int(center_x_anchor - 20), int(center_y_anchor - 20), 40, 40

            confidences.append(conf)

            # Compute head kinetic pose proxies (pitch & yaw in degrees)
            face_center_x = x + (w / 2.0)
            face_center_y = y + (h / 2.0)
            yaw_deg = round(float(((face_center_x - center_x_anchor) / w_frame) * 90.0), 2)
            pitch_deg = round(float(((face_center_y - center_y_anchor) / h_frame) * 90.0), 2)
            head_yaw_series.append(yaw_deg)
            head_pitch_series.append(pitch_deg)

            # Compute mouth dynamics (lower 35% of face region)
            mouth_y1 = max(0, y + int(0.65 * h))
            mouth_y2 = min(gray.shape[0], y + h)
            mouth_x1 = max(0, x + int(0.18 * w))
            mouth_x2 = min(gray.shape[1], x + int(0.82 * w))
            mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

            mouth_motion_score = 0.05
            if mouth_roi.size > 0:
                resized_mouth = cv2.resize(mouth_roi, (32, 24))
                if prev_mouth_roi is not None:
                    diff = np.mean(np.abs(resized_mouth.astype(float) - prev_mouth_roi.astype(float)))
                    mouth_motion_score = min(0.95, max(0.04, float(diff / 25.0)))
                prev_mouth_roi = resized_mouth.copy()

            mouth_series.append(mouth_motion_score)

            frames_metadata.append(
                {
                    "frame_idx": processed_idx,
                    "timestamp_sec": timestamp_sec,
                    "face_detected": face_detected,
                    "face_confidence": round(conf, 3),
                    "mean_luminance": round(lum, 1),
                    "head_pitch": pitch_deg,
                    "head_yaw": yaw_deg,
                    "eye_blink": False,
                    "mouth_openness": round(mouth_motion_score, 4),
                }
            )

            frame_idx += 1
            processed_idx += 1

        cap.release()

        # Aggregate statistics
        total_frames = len(frames_metadata)
        if total_frames == 0:
            return {
                "total_frames": 0,
                "fps": effective_fps,
                "duration_sec": 0.0,
                "mean_face_confidence": 0.0,
                "face_detection_ratio": 0.0,
                "mean_luminance": 0.0,
                "head_pitch_variance": 0.0,
                "head_yaw_variance": 0.0,
                "mouth_openness_variance": 0.0,
                "mouth_series": np.zeros(1),
                "frames": [],
            }

        det_ratio = float(sum(1 for f in frames_metadata if f["face_detected"]) / total_frames)
        mean_conf = float(np.mean(confidences))
        mean_lum = float(np.mean(luminances))
        pitch_var = float(np.var(head_pitch_series)) if len(head_pitch_series) > 1 else 0.0
        yaw_var = float(np.var(head_yaw_series)) if len(head_yaw_series) > 1 else 0.0
        mouth_var = float(np.var(mouth_series)) if len(mouth_series) > 1 else 0.0

        return {
            "total_frames": total_frames,
            "fps": round(effective_fps, 1),
            "duration_sec": round(total_frames / effective_fps, 2),
            "mean_face_confidence": round(mean_conf, 3),
            "face_detection_ratio": round(det_ratio, 3),
            "mean_luminance": round(mean_lum, 1),
            "head_pitch_variance": round(pitch_var, 2),
            "head_yaw_variance": round(yaw_var, 2),
            "mouth_openness_variance": round(mouth_var, 4),
            "mouth_series": np.array(mouth_series),
            "frames": frames_metadata,
        }
