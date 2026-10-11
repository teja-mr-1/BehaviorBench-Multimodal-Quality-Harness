"""Unit tests for multimodal signal extraction."""

import numpy as np
import pytest
from behaviorbench.config import QualityThresholds
from behaviorbench.signal_extractor import (
    estimate_av_sync_drift,
    extract_audio_features,
    extract_video_features,
)


def test_clipping_detection():
    # Audio with 20% clipped samples
    samples = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    samples[samples > 0.8] = 0.98

    clipped_count = np.sum(np.abs(samples) >= 0.96)
    ratio = float(clipped_count / len(samples))
    assert ratio > 0.15


def test_estimate_av_sync_drift_zero():
    # Perfectly aligned signals
    audio_rms = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1], dtype=np.float32)
    mouth_series = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1], dtype=np.float32)

    drift = estimate_av_sync_drift(
        audio_rms=audio_rms, mouth_series=mouth_series, video_fps=30
    )
    assert abs(drift) < 50.0


def test_video_feature_extraction_empty():
    res = extract_video_features([])
    assert res["total_frames"] == 0
    assert res["mean_face_confidence"] == 0.0
