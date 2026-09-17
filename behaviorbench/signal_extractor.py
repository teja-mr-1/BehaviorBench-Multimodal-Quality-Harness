"""Multimodal signal extraction module for BehaviorBench.

Extracts acoustic features (pitch, pause/hesitation duration, clipping, SNR)
and visual features (head pose variance, luminance, face confidence, blinks),
along with audio-visual synchronization drift estimation.
"""

import json
from typing import List, Optional, Tuple, Union
import wave
import numpy as np
from scipy.signal import correlate
from behaviorbench.config import BehavioralFeatures, QualityThresholds


def extract_audio_features(wav_path: str, thresholds: Optional[QualityThresholds] = None) -> dict:
    """Extract acoustic features and voice activity metrics from WAV audio."""
    if thresholds is None:
        thresholds = QualityThresholds()

    with wave.open(wav_path, "r") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    duration_sec = n_frames / rate if rate > 0 else 0.0

    # Unpack to float32 between -1.0 and 1.0
    if sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if len(samples) == 0:
        return {
            "duration_sec": 0.0,
            "speech_duration_sec": 0.0,
            "silence_ratio": 1.0,
            "hesitation_pauses_count": 0,
            "mean_pitch_hz": 0.0,
            "pitch_variance": 0.0,
            "clipping_ratio": 0.0,
            "snr_db": 0.0,
            "rms_envelope": np.zeros(1),
            "sample_rate": rate,
            "hop_len": 400,
            "speech_intervals": []
        }

    # 1. Audio clipping detection (samples at +/- 0.96 or higher)
    clipped_count = np.sum(np.abs(samples) >= 0.96)
    clipping_ratio = float(clipped_count / len(samples))

    # 2. Frame-level RMS energy calculation (50ms window with 25ms hop)
    frame_len = max(1, int(0.050 * rate))
    hop_len = max(1, int(0.025 * rate))
    n_energy_frames = max(1, (len(samples) - frame_len) // hop_len + 1)

    rms_envelope = []
    for i in range(n_energy_frames):
        start = i * hop_len
        window = samples[start : start + frame_len]
        rms = np.sqrt(np.mean(window**2) + 1e-9)
        rms_envelope.append(rms)
    rms_envelope = np.array(rms_envelope)

    # 3. Voice Activity & Hesitation Detection
    noise_floor = np.percentile(rms_envelope, 15)
    max_energy = np.max(rms_envelope) if len(rms_envelope) > 0 else 1.0
    energy_threshold = max(0.015, noise_floor + 0.15 * (max_energy - noise_floor))
    is_speech_frame = rms_envelope > energy_threshold

    speech_frames_count = np.sum(is_speech_frame)
    speech_duration_sec = float((speech_frames_count * hop_len) / rate)
    silence_ratio = float(max(0.0, 1.0 - (speech_duration_sec / max(0.1, duration_sec))))

    # Extract speech intervals
    speech_intervals: List[Tuple[float, float]] = []
    in_speech = False
    start_time = 0.0

    for idx, active in enumerate(is_speech_frame):
        t = (idx * hop_len) / rate
        if active and not in_speech:
            in_speech = True
            start_time = t
        elif not active and in_speech:
            in_speech = False
            speech_intervals.append((round(start_time, 2), round(t, 2)))
    if in_speech:
        speech_intervals.append((round(start_time, 2), round(duration_sec, 2)))

    # Count distinct pauses >= hesitation threshold
    hesitation_count = 0
    in_pause = False
    pause_len_frames = 0
    min_pause_frames = int(thresholds.hesitation_pause_threshold_sec / (hop_len / rate))

    for active in is_speech_frame:
        if not active:
            pause_len_frames += 1
            in_pause = True
        else:
            if in_pause and pause_len_frames >= min_pause_frames:
                hesitation_count += 1
            in_pause = False
            pause_len_frames = 0
    if in_pause and pause_len_frames >= min_pause_frames:
        hesitation_count += 1

    # 4. Signal-to-Noise Ratio (SNR) in dB
    speech_energy = np.mean(rms_envelope[is_speech_frame] ** 2) if np.any(is_speech_frame) else 1e-6
    silence_energy = np.mean(rms_envelope[~is_speech_frame] ** 2) if np.any(~is_speech_frame) else 1e-6
    if silence_energy < 1e-5:
        snr_db = 28.5  # Pristine digital baseline
    else:
        snr_db = float(10.0 * np.log10(max(1.0, speech_energy / silence_energy)))

    # 5. Fundamental Frequency (F0) estimate via autocorrelation on speech
    pitches = []
    for i in range(0, len(samples) - 1024, 2048):
        chunk = samples[i : i + 1024]
        if np.std(chunk) > 0.05:
            corr = np.correlate(chunk, chunk, mode="full")
            corr = corr[len(corr) // 2 :]
            min_lag = int(rate / 350)
            max_lag = int(rate / 50)
            if max_lag < len(corr):
                peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                if corr[peak_lag] > 0.3 * corr[0]:
                    pitches.append(rate / peak_lag)

    mean_pitch = float(np.mean(pitches)) if pitches else 180.0
    pitch_variance = float(np.var(pitches)) if len(pitches) > 1 else 10.0

    return {
        "duration_sec": round(duration_sec, 2),
        "speech_duration_sec": round(speech_duration_sec, 2),
        "silence_ratio": round(silence_ratio, 3),
        "hesitation_pauses_count": hesitation_count,
        "mean_pitch_hz": round(mean_pitch, 1),
        "pitch_variance": round(pitch_variance, 1),
        "clipping_ratio": round(clipping_ratio, 4),
        "snr_db": round(snr_db, 1),
        "rms_envelope": rms_envelope,
        "sample_rate": rate,
        "hop_len": hop_len,
        "speech_intervals": speech_intervals,
    }


def extract_video_features(landmarks_source: Union[str, dict, list]) -> dict:
    """Extract visual behavioral features and frame stats from landmark JSON or metadata list."""
    if isinstance(landmarks_source, str):
        with open(landmarks_source, "r") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        fps = data.get("fps", 30)
    elif isinstance(landmarks_source, dict):
        frames = landmarks_source.get("frames", [])
        fps = landmarks_source.get("fps", 30)
    elif isinstance(landmarks_source, list):
        frames = landmarks_source
        fps = 30
    else:
        frames = []
        fps = 30

    total_frames = len(frames)
    if total_frames == 0:
        return {
            "total_frames": 0,
            "mean_face_confidence": 0.0,
            "mean_luminance": 0.0,
            "head_pitch_variance": 0.0,
            "head_yaw_variance": 0.0,
            "blink_count": 0,
            "mouth_openness_variance": 0.0,
            "mouth_series": np.zeros(1),
            "fps": fps,
        }

    confidences = [
        f.get("face_confidence", 0.95)
        for f in frames
        if f.get("face_detected", True)
    ]
    luminances = [f.get("luminance", f.get("mean_luminance", 100.0)) for f in frames]
    pitches = [f.get("head_pitch_deg", f.get("head_pitch", 0.0)) for f in frames]
    yaws = [f.get("head_yaw_deg", f.get("head_yaw", 0.0)) for f in frames]
    mouths = [f.get("mouth_openness", 0.05) for f in frames]
    blinks = sum(1 for f in frames if f.get("is_blink", f.get("eye_blink", False)))

    return {
        "total_frames": total_frames,
        "mean_face_confidence": round(
            float(np.mean(confidences)) if confidences else 0.95, 3
        ),
        "mean_luminance": round(float(np.mean(luminances)), 1),
        "head_pitch_variance": round(float(np.var(pitches)), 2),
        "head_yaw_variance": round(float(np.var(yaws)), 2),
        "blink_count": blinks,
        "mouth_openness_variance": round(float(np.var(mouths)), 4),
        "mouth_series": np.array(mouths),
        "fps": fps,
    }


def estimate_av_sync_drift(
    audio_rms: np.ndarray,
    audio_hop_len: int = 400,
    audio_rate: int = 16000,
    mouth_series: Optional[np.ndarray] = None,
    video_fps: int = 30,
) -> float:
    """Estimate Audio-Video synchronization drift in milliseconds via cross-correlation."""
    if mouth_series is None or len(mouth_series) == 0 or len(audio_rms) == 0:
        return 0.0

    # Resample audio RMS envelope to match video FPS (30Hz)
    resampled_len = int(len(mouth_series))
    if resampled_len <= 1:
        return 0.0

    t_audio = np.linspace(0, 1, len(audio_rms))
    t_video = np.linspace(0, 1, resampled_len)
    resampled_audio = np.interp(t_video, t_audio, audio_rms)

    # Normalize both signals
    sig1 = resampled_audio - np.mean(resampled_audio)
    std1 = np.std(sig1)
    if std1 > 0:
        sig1 /= std1

    sig2 = mouth_series - np.mean(mouth_series)
    std2 = np.std(sig2)
    if std2 < 0.01:
        return 0.0  # Mouth movement was static or untracked
    sig2 /= std2

    # Cross-correlation within +/- 500ms lag window (+/- 15 frames at 30fps)
    max_lag_frames = int(0.500 * video_fps)
    corr = correlate(sig1, sig2, mode="full")
    mid_idx = len(corr) // 2
    window = corr[max(0, mid_idx - max_lag_frames) : min(len(corr), mid_idx + max_lag_frames + 1)]

    if len(window) == 0:
        return 0.0

    peak_offset = np.argmax(window) - max_lag_frames
    drift_ms = float((peak_offset / video_fps) * 1000.0)
    return round(drift_ms, 1)


def extract_multimodal_features(
    wav_path: str, landmarks_path: str, thresholds: Optional[QualityThresholds] = None
) -> BehavioralFeatures:
    """Extract complete multimodal feature set for a conversation clip."""
    if thresholds is None:
        thresholds = QualityThresholds()

    a_feat = extract_audio_features(wav_path, thresholds)
    v_feat = extract_video_features(landmarks_path)

    drift_ms = estimate_av_sync_drift(
        audio_rms=a_feat["rms_envelope"],
        audio_hop_len=a_feat["hop_len"],
        audio_rate=a_feat["sample_rate"],
        mouth_series=v_feat["mouth_series"],
        video_fps=v_feat["fps"],
    )

    return BehavioralFeatures(
        duration_sec=a_feat["duration_sec"],
        speech_duration_sec=a_feat["speech_duration_sec"],
        silence_ratio=a_feat["silence_ratio"],
        hesitation_pauses_count=a_feat["hesitation_pauses_count"],
        mean_pitch_hz=a_feat["mean_pitch_hz"],
        pitch_variance=a_feat["pitch_variance"],
        clipping_ratio=a_feat["clipping_ratio"],
        snr_db=a_feat["snr_db"],
        total_frames=v_feat["total_frames"],
        mean_face_confidence=v_feat["mean_face_confidence"],
        mean_luminance=v_feat["mean_luminance"],
        head_pitch_variance=v_feat["head_pitch_variance"],
        head_yaw_variance=v_feat["head_yaw_variance"],
        blink_count=v_feat["blink_count"],
        mouth_openness_variance=v_feat["mouth_openness_variance"],
        av_sync_drift_ms=drift_ms,
    )
