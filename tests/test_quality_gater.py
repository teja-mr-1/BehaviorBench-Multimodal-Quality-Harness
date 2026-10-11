"""Unit tests for automated quality gater and social signal categorization."""

import pytest
from behaviorbench.config import (
    BehavioralFeatures,
    ClipStatus,
    QualityThresholds,
    SocialSignalCategory,
)
from behaviorbench.quality_gater import evaluate_clip_quality


@pytest.fixture
def base_features():
    return BehavioralFeatures(
        duration_sec=5.0,
        speech_duration_sec=4.0,
        silence_ratio=0.20,
        hesitation_pauses_count=0,
        mean_pitch_hz=160.0,
        pitch_variance=15.0,
        clipping_ratio=0.001,
        snr_db=22.0,
        zero_crossing_rate=0.04,
        total_frames=150,
        mean_face_confidence=0.95,
        mean_luminance=120.0,
        head_pitch_variance=1.5,
        head_yaw_variance=2.0,
        blink_count=2,
        mouth_openness_variance=0.04,
        av_sync_drift_ms=0.0,
    )


def test_clean_baseline(base_features):
    thresholds = QualityThresholds()
    res = evaluate_clip_quality("clean_test", base_features, thresholds)
    assert res.status == ClipStatus.PASS_CLEAN
    assert res.benchmark_slice == "clean_baseline"
    assert res.social_signal == SocialSignalCategory.CLEAN_BASELINE
    assert res.quality_score == 100.0


def test_rejection_on_excessive_clipping(base_features):
    base_features.clipping_ratio = 0.05  # 5% clipping > 1.5% limit
    thresholds = QualityThresholds()
    res = evaluate_clip_quality("clip_test", base_features, thresholds)
    assert res.status == ClipStatus.REJECTED
    assert res.social_signal == SocialSignalCategory.ACOUSTIC_DISTORTION
    assert len(res.anomalies_detected) > 0


def test_rejection_on_excessive_sync_drift(base_features):
    base_features.av_sync_drift_ms = 180.0  # > 120ms limit
    thresholds = QualityThresholds()
    res = evaluate_clip_quality("sync_test", base_features, thresholds)
    assert res.status == ClipStatus.REJECTED
    assert res.social_signal == SocialSignalCategory.AV_DESYNCHRONIZATION


def test_gold_slice_hesitation(base_features):
    base_features.hesitation_pauses_count = 2
    thresholds = QualityThresholds()
    res = evaluate_clip_quality("hesitant_test", base_features, thresholds)
    assert res.status == ClipStatus.PASS_EDGE_CASE
    assert res.benchmark_slice == "gold_slice_hesitation"
    assert res.social_signal == SocialSignalCategory.HESITATION_UNCERTAINTY


def test_gold_slice_agreement_nodding(base_features):
    base_features.head_pitch_variance = 8.5  # nodding
    thresholds = QualityThresholds()
    res = evaluate_clip_quality("nodding_test", base_features, thresholds)
    assert res.status == ClipStatus.PASS_EDGE_CASE
    assert res.benchmark_slice == "gold_slice_agreement"
    assert res.social_signal == SocialSignalCategory.KINETIC_AGREEMENT
