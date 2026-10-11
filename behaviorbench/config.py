"""Pydantic configuration schemas and validation thresholds for BehaviorBench 2.0.

Aligned with Interhuman.ai multimodal requirements:
- Audio clipping and true SNR gatekeeping
- Head kinetics & facial landmark dynamics
- Sub-50ms AV cross-correlation sync detection
- Targeted social signal benchmark slicing (Hesitation, Agreement, Reflection)
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClipStatus(str, Enum):
    PASS_CLEAN = "PASS_CLEAN"
    PASS_EDGE_CASE = "PASS_EDGE_CASE"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class SocialSignalCategory(str, Enum):
    """Interhuman Inter-2 aligned behavioral and conversational taxonomy."""
    CLEAN_BASELINE = "clean_baseline"
    HESITATION_UNCERTAINTY = "hesitation_uncertainty"
    KINETIC_AGREEMENT = "kinetic_agreement"
    EXPRESSIVE_PROSODY = "expressive_prosody"
    COGNITIVE_REFLECTION = "cognitive_reflection"
    ACOUSTIC_DISTORTION = "acoustic_distortion"
    AV_DESYNCHRONIZATION = "av_desynchronization"
    UNDEREXPOSED_LIGHTING = "underexposed_lighting"


class QualityThresholds(BaseModel):
    """Thresholds for audio, video, and synchronization quality gating."""

    # Audio thresholds
    max_clipping_ratio: float = Field(
        0.015, description="Max allowed ratio of clipped audio samples (1.5%)"
    )
    min_snr_db: float = Field(
        12.0, description="Minimum Signal-to-Noise Ratio in decibels"
    )
    min_speech_ratio: float = Field(
        0.10, description="Minimum speech active duration ratio"
    )

    # Video thresholds
    min_face_confidence: float = Field(
        0.70, description="Minimum face detection confidence"
    )
    min_luminance: float = Field(
        30.0, description="Minimum average frame luminance (0-255 scale)"
    )
    max_frame_drop_ratio: float = Field(
        0.05, description="Max allowed percentage of dropped frames"
    )

    # Synchronization thresholds
    max_av_drift_ms: float = Field(
        120.0,
        description="Max allowable Audio-Video sync drift in milliseconds (relaxed)",
    )
    strict_av_drift_ms: float = Field(
        45.0,
        description="Strict Interhuman Inter-2 sync drift threshold in milliseconds",
    )

    # Behavioral classification thresholds
    hesitation_pause_threshold_sec: float = Field(
        0.80, description="Pause duration in seconds flagged as hesitation"
    )
    high_head_movement_threshold_deg: float = Field(
        18.0, description="Head pose angle variance threshold for lateral turn"
    )
    expressive_pitch_variance_hz: float = Field(
        30.0, description="Standard deviation in F0 pitch for expressive prosody"
    )

    @classmethod
    def interhuman_preset(cls) -> "QualityThresholds":
        """Strict configuration optimized for Interhuman Inter-2 training curation."""
        return cls(
            max_clipping_ratio=0.008,  # 0.8% strict limit
            min_snr_db=14.0,           # High clarity
            min_luminance=35.0,
            max_av_drift_ms=60.0,      # Strict AV sync to eliminate hallucinations
            strict_av_drift_ms=35.0,
            hesitation_pause_threshold_sec=0.65,
        )

    @classmethod
    def zero_hallucination_preset(cls) -> "QualityThresholds":
        """Ultra-strict gate to purge any corrupted timestamps or phantom audio."""
        return cls(
            max_clipping_ratio=0.005,
            min_snr_db=16.0,
            min_luminance=40.0,
            max_av_drift_ms=45.0,
            strict_av_drift_ms=30.0,
            hesitation_pause_threshold_sec=0.50,
        )


class BehavioralFeatures(BaseModel):
    """Extracted acoustic and facial behavioral metrics."""

    # Audio metrics
    duration_sec: float
    speech_duration_sec: float
    silence_ratio: float
    hesitation_pauses_count: int
    mean_pitch_hz: float
    pitch_variance: float
    clipping_ratio: float
    snr_db: float
    zero_crossing_rate: float = Field(
        default=0.05, description="Mean audio zero crossing rate"
    )

    # Video metrics
    total_frames: int
    mean_face_confidence: float
    mean_luminance: float
    head_pitch_variance: float
    head_yaw_variance: float
    blink_count: int
    mouth_openness_variance: float

    # Multimodal synchronization
    av_sync_drift_ms: float


class ClipInspectionResult(BaseModel):
    """Complete inspection output for a single multimodal clip."""

    clip_id: str
    status: ClipStatus
    quality_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall quality score from 0 to 100"
    )
    anomalies_detected: List[str] = Field(default_factory=list)
    features: BehavioralFeatures
    benchmark_slice: Optional[str] = Field(
        None, description="Partition split e.g. 'clean_baseline' or 'gold_slice_edge_case'"
    )
    social_signal: Optional[SocialSignalCategory] = Field(
        None, description="Inter-2 social signal classification"
    )
