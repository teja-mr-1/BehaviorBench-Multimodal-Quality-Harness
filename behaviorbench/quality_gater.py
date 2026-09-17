"""Automated multimodal quality gatekeeper module for BehaviorBench.

Evaluates extracted acoustic, visual and synchronization metrics against thresholds,
flags anomalies, and assigns quality scores and validation statuses.
"""

from typing import List, Tuple
from behaviorbench.config import (
    BehavioralFeatures,
    ClipInspectionResult,
    ClipStatus,
    QualityThresholds,
)


def evaluate_clip_quality(
    clip_id: str,
    features: BehavioralFeatures,
    thresholds: QualityThresholds
) -> ClipInspectionResult:
    """Evaluate quality gates and classify clip into Clean, Edge-Case, Flagged, or Rejected."""
    anomalies: List[str] = []
    penalty = 0.0

    # 1. Audio Quality Checks
    if features.clipping_ratio > thresholds.max_clipping_ratio:
        anomalies.append(f"Audio clipping detected ({features.clipping_ratio*100:.1f}% > {thresholds.max_clipping_ratio*100:.1f}%)")
        penalty += 45.0

    if features.snr_db < thresholds.min_snr_db:
        anomalies.append(f"Low Signal-to-Noise Ratio ({features.snr_db:.1f} dB < {thresholds.min_snr_db:.1f} dB)")
        penalty += 25.0

    # 2. Video Quality Checks
    if features.mean_luminance < thresholds.min_luminance:
        anomalies.append(f"Underexposed lighting ({features.mean_luminance:.1f} < {thresholds.min_luminance:.1f} lum)")
        penalty += 35.0

    if features.mean_face_confidence < thresholds.min_face_confidence:
        anomalies.append(f"Low face detection confidence ({features.mean_face_confidence:.2f} < {thresholds.min_face_confidence:.2f})")
        penalty += 30.0

    # 3. Audio-Visual Synchronization Checks
    if abs(features.av_sync_drift_ms) > thresholds.max_av_drift_ms:
        anomalies.append(f"AV sync drift exceeded ({abs(features.av_sync_drift_ms):.1f}ms > {thresholds.max_av_drift_ms:.1f}ms)")
        penalty += 40.0

    # Calculate final Quality Score (0 to 100)
    raw_score = 100.0 - penalty
    quality_score = max(0.0, min(100.0, raw_score))

    # Determine Status & Benchmark Slice
    if penalty >= 40.0:
        status = ClipStatus.REJECTED
        benchmark_slice = None
    elif penalty > 0.0:
        status = ClipStatus.FLAGGED
        benchmark_slice = None
    else:
        # Check if it exhibits edge-case behavioral variation
        is_hesitant = features.hesitation_pauses_count > 0
        is_large_head_turn = features.head_yaw_variance > thresholds.high_head_movement_threshold_deg
        is_expressive_nod = features.head_pitch_variance > 6.0

        if is_hesitant or is_large_head_turn or is_expressive_nod:
            status = ClipStatus.PASS_EDGE_CASE
            benchmark_slice = "gold_slice_edge_case"
        else:
            status = ClipStatus.PASS_CLEAN
            benchmark_slice = "clean_baseline"

    return ClipInspectionResult(
        clip_id=clip_id,
        status=status,
        quality_score=round(quality_score, 1),
        anomalies_detected=anomalies,
        features=features,
        benchmark_slice=benchmark_slice
    )
