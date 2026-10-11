"""Unit tests for OpenCV vision tracker on real video."""

import os
import pytest
from behaviorbench.vision_tracker import VideoVisionTracker


def test_vision_tracker_on_sample_video():
    sample_mp4 = "sample_data/real_interview_male_researcher.mp4"
    if not os.path.exists(sample_mp4):
        pytest.skip("Sample MP4 video not found")

    tracker = VideoVisionTracker()
    results = tracker.track_video(sample_mp4, target_fps=30)

    assert results["total_frames"] > 50
    assert results["fps"] > 0
    assert results["mean_face_confidence"] > 0.70
    assert results["face_detection_ratio"] > 0.80
    assert len(results["mouth_series"]) == results["total_frames"]
    assert results["mouth_openness_variance"] > 0.0
