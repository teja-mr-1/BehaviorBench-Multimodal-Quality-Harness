"""Pydantic configuration schemas and validation thresholds for BehaviorBench."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ClipStatus(str, Enum):
    PASS_CLEAN = "PASS_CLEAN"
    PASS_EDGE_CASE = "PASS_EDGE_CASE"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class QualityThresholds(BaseModel):
    """Thresholds for audio, video, and synchronization quality gating."""
    
    # Audio thresholds
    max_clipping_ratio: float = Field(0.015, description="Max allowed ratio of clipped audio samples (1.5%)")
    min_snr_db: float = Field(12.0, description="Minimum Signal-to-Noise Ratio in decibels")
    min_speech_ratio: float = Field(0.10, description="Minimum speech active duration ratio")
    
    # Video thresholds
    min_face_confidence: float = Field(0.70, description="Minimum face detection confidence")
    min_luminance: float = Field(30.0, description="Minimum average frame luminance (0-255 scale)")
    max_frame_drop_ratio: float = Field(0.05, description="Max allowed percentage of dropped frames")
    
    # Synchronization thresholds
    max_av_drift_ms: float = Field(120.0, description="Max allowable Audio-Video sync drift in milliseconds")
    
    # Behavioral classification thresholds
    hesitation_pause_threshold_sec: float = Field(0.80, description="Pause duration in seconds flagged as hesitation")
    high_head_movement_threshold_deg: float = Field(18.0, description="Head pose angle variance threshold for edge-case")


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
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score from 0 to 100")
    anomalies_detected: List[str] = Field(default_factory=list)
    features: BehavioralFeatures
    benchmark_slice: Optional[str] = Field(None, description="'clean_baseline' or 'gold_slice_edge_case' or None")
