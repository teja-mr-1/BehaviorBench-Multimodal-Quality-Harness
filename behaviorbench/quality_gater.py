"""Automated multimodal quality gatekeeper module for BehaviorBench 2.0.

Evaluates acoustic, visual and synchronization metrics against thresholds,
flags anomalies, assigns quality scores, and categorizes clips into
Interhuman Inter-2 aligned social signal benchmark slices.
"""

from typing import List, Tuple
from behaviorbench.config import (
    BehavioralFeatures,
    ClipInspectionResult,
    ClipStatus,
    QualityThresholds,
    SocialSignalCategory,
)


def evaluate_clip_quality(
    clip_id: str,
    features: BehavioralFeatures,
    thresholds: QualityThresholds,
) -> ClipInspectionResult:
    """Evaluate quality gates and classify clip into Clean, Edge-Case, Flagged, or Rejected."""
    anomalies: List[str] = []
    penalty = 0.0

    # 1. Audio Quality Checks
    if features.clipping_ratio > thresholds.max_clipping_ratio:
        anomalies.append(
            f"Audio clipping detected ({features.clipping_ratio*100:.1f}% > {thresholds.max_clipping_ratio*100:.1f}%)"
        )
        penalty += 45.0

    if features.snr_db < thresholds.min_snr_db:
        anomalies.append(
            f"Low Signal-to-Noise Ratio ({features.snr_db:.1f} dB < {thresholds.min_snr_db:.1f} dB)"
        )
        penalty += 25.0

    # 2. Video Quality Checks
    if features.mean_luminance < thresholds.min_luminance:
        anomalies.append(
            f"Underexposed lighting ({features.mean_luminance:.1f} < {thresholds.min_luminance:.1f} lum)"
        )
        penalty += 35.0

    if features.mean_face_confidence < thresholds.min_face_confidence:
        anomalies.append(
            f"Low face detection confidence ({features.mean_face_confidence:.2f} < {thresholds.min_face_confidence:.2f})"
        )
        penalty += 30.0

    # 3. Audio-Visual Synchronization Checks
    drift_mag = abs(features.av_sync_drift_ms)
    if drift_mag > thresholds.max_av_drift_ms:
        anomalies.append(
            f"AV sync drift exceeded ({drift_mag:.1f}ms > {thresholds.max_av_drift_ms:.1f}ms)"
        )
        penalty += 40.0
    elif drift_mag > thresholds.strict_av_drift_ms:
        anomalies.append(
            f"Marginal AV sync drift ({drift_mag:.1f}ms > {thresholds.strict_av_drift_ms:.1f}ms strict)"
        )
        penalty += 15.0

    # Calculate final Quality Score (0 to 100)
    raw_score = 100.0 - penalty
    quality_score = max(0.0, min(100.0, raw_score))

    # Determine Status, Benchmark Slice & Social Signal Category
    if penalty >= 40.0:
        status = ClipStatus.REJECTED
        benchmark_slice = None
        if drift_mag > thresholds.max_av_drift_ms:
            social_signal = SocialSignalCategory.AV_DESYNCHRONIZATION
        elif features.clipping_ratio > thresholds.max_clipping_ratio:
            social_signal = SocialSignalCategory.ACOUSTIC_DISTORTION
        elif features.mean_luminance < thresholds.min_luminance:
            social_signal = SocialSignalCategory.UNDEREXPOSED_LIGHTING
        else:
            social_signal = SocialSignalCategory.ACOUSTIC_DISTORTION

    elif penalty > 0.0:
        status = ClipStatus.FLAGGED
        benchmark_slice = "quarantined_review"
        if drift_mag > thresholds.strict_av_drift_ms:
            social_signal = SocialSignalCategory.AV_DESYNCHRONIZATION
        elif features.snr_db < thresholds.min_snr_db:
            social_signal = SocialSignalCategory.ACOUSTIC_DISTORTION
        else:
            social_signal = SocialSignalCategory.UNDEREXPOSED_LIGHTING

    else:
        # Candidate passed all gates - check behavioral variations for Gold Slice splits
        is_hesitant = features.hesitation_pauses_count > 0
        is_nodding = features.head_pitch_variance > 5.0
        is_large_head_turn = (
            features.head_yaw_variance > thresholds.high_head_movement_threshold_deg
        )
        is_expressive = features.pitch_variance > thresholds.expressive_pitch_variance_hz

        if is_hesitant:
            status = ClipStatus.PASS_EDGE_CASE
            benchmark_slice = "gold_slice_hesitation"
            social_signal = SocialSignalCategory.HESITATION_UNCERTAINTY
        elif is_nodding:
            status = ClipStatus.PASS_EDGE_CASE
            benchmark_slice = "gold_slice_agreement"
            social_signal = SocialSignalCategory.KINETIC_AGREEMENT
        elif is_expressive:
            status = ClipStatus.PASS_EDGE_CASE
            benchmark_slice = "gold_slice_expressive"
            social_signal = SocialSignalCategory.EXPRESSIVE_PROSODY
        elif is_large_head_turn:
            status = ClipStatus.PASS_EDGE_CASE
            benchmark_slice = "gold_slice_reflection"
            social_signal = SocialSignalCategory.COGNITIVE_REFLECTION
        else:
            status = ClipStatus.PASS_CLEAN
            benchmark_slice = "clean_baseline"
            social_signal = SocialSignalCategory.CLEAN_BASELINE

    return ClipInspectionResult(
        clip_id=clip_id,
        status=status,
        quality_score=round(quality_score, 1),
        anomalies_detected=anomalies,
        features=features,
        benchmark_slice=benchmark_slice,
        social_signal=social_signal,
    )
