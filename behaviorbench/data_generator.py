"""Synthetic conversational multimodal data generator for BehaviorBench.

Generates realistic audio WAV files and synchronized 30fps facial landmark frames
with various natural conversational patterns, subtle edge cases, and corruptions.
"""

import json
import math
import os
import wave
import numpy as np


def generate_audio_wav(
    file_path: str,
    duration_sec: float = 4.0,
    sample_rate: int = 16000,
    pitch_hz: float = 180.0,
    speech_segments: list = None,
    add_noise: bool = False,
    noise_level: float = 0.02,
    clip_audio: bool = False,
    clipping_threshold: float = 0.95
):
    """Generate a synthetic conversational speech WAV file."""
    if speech_segments is None:
        speech_segments = [(0.5, 1.8), (2.4, 3.8)]

    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    audio = np.zeros(total_samples, dtype=np.float32)

    for start_sec, end_sec in speech_segments:
        s_idx = int(start_sec * sample_rate)
        e_idx = int(end_sec * sample_rate)
        if s_idx >= total_samples or e_idx > total_samples:
            continue
        seg_t = t[s_idx:e_idx]
        
        # Synthesize harmonic voice formant
        fundamental = 0.6 * np.sin(2 * np.pi * pitch_hz * seg_t)
        second_harmonic = 0.25 * np.sin(2 * np.pi * (pitch_hz * 2.1) * seg_t)
        third_harmonic = 0.15 * np.sin(2 * np.pi * (pitch_hz * 3.0) * seg_t)
        envelope = np.sin(np.linspace(0, np.pi, len(seg_t))) ** 0.5
        audio[s_idx:e_idx] = (fundamental + second_harmonic + third_harmonic) * envelope * 0.7

    if add_noise:
        noise = np.random.normal(0, noise_level, total_samples)
        audio += noise

    if clip_audio:
        # Amplify and hard clip at digital ceiling
        audio = np.clip(audio * 4.0, -1.0, 1.0)
    else:
        # Normalize to safe amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.75

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with wave.open(file_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def generate_video_landmarks(
    file_path: str,
    duration_sec: float = 4.0,
    fps: int = 30,
    speech_segments: list = None,
    mean_luminance: float = 120.0,
    face_confidence: float = 0.96,
    nod_frequency_hz: float = 0.0,
    turn_away_deg: float = 0.0,
    av_lag_sec: float = 0.0
):
    """Generate synchronized 30fps facial landmark metadata JSON."""
    if speech_segments is None:
        speech_segments = [(0.5, 1.8), (2.4, 3.8)]

    total_frames = int(duration_sec * fps)
    frames_data = []

    for frame_idx in range(total_frames):
        timestamp_sec = frame_idx / fps
        adjusted_time = max(0.0, timestamp_sec - av_lag_sec)

        # Check if mouth should be open (speech active with lag)
        is_speaking = any(s <= adjusted_time <= e for s, e in speech_segments)
        
        # Mouth openness (0.0 closed to 1.0 wide open)
        if is_speaking:
            mouth_openness = 0.35 + 0.25 * math.sin(2 * math.pi * 4.5 * timestamp_sec)
        else:
            mouth_openness = 0.05 + 0.03 * math.sin(2 * math.pi * 1.0 * timestamp_sec)

        # Head pose
        if nod_frequency_hz > 1.0:
            pitch_deg = 4.5 * math.sin(2 * math.pi * nod_frequency_hz * timestamp_sec) # Expressive nod
        elif nod_frequency_hz > 0:
            pitch_deg = 0.8 * math.sin(2 * math.pi * nod_frequency_hz * timestamp_sec) # Subtle natural drift
        else:
            pitch_deg = 0.0
        yaw_deg = turn_away_deg * (0.5 + 0.5 * math.sin(2 * math.pi * 0.4 * timestamp_sec)) if turn_away_deg > 0 else 0.0
        roll_deg = 0.5 * math.sin(2 * math.pi * 0.2 * timestamp_sec)

        # Blink simulation (every ~3 seconds)
        blink = (frame_idx % 90) in [45, 46, 47]

        frames_data.append({
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "face_detected": True,
            "face_confidence": round(face_confidence + np.random.uniform(-0.02, 0.02), 3),
            "luminance": round(mean_luminance + np.random.uniform(-3, 3), 1),
            "mouth_openness": round(max(0.0, mouth_openness), 3),
            "head_pitch_deg": round(pitch_deg, 2),
            "head_yaw_deg": round(yaw_deg, 2),
            "head_roll_deg": round(roll_deg, 2),
            "is_blink": blink
        })

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump({
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "frames": frames_data
        }, f, indent=2)


def generate_synthetic_dataset(output_dir: str):
    """Generate 10 distinct multimodal conversation test clips."""
    os.makedirs(output_dir, exist_ok=True)
    
    clips_config = [
        {
            "id": "clip_01_clean_active",
            "desc": "Clean neutral conversation, balanced tone and steady head pose",
            "duration": 4.0,
            "speech": [(0.4, 1.8), (2.2, 3.8)],
            "pitch": 190.0,
            "noise": False,
            "clip_audio": False,
            "lum": 135.0,
            "conf": 0.98,
            "nod": 0.2,
            "turn": 2.0,
            "lag": 0.0
        },
        {
            "id": "clip_02_thoughtful_hesitant",
            "desc": "Long vocal pause (hesitation 1.4s), reflective head tilt (Gold Slice Edge Case)",
            "duration": 5.0,
            "speech": [(0.4, 1.2), (2.6, 4.6)], # 1.4s hesitation gap
            "pitch": 175.0,
            "noise": False,
            "clip_audio": False,
            "lum": 125.0,
            "conf": 0.94,
            "nod": 0.5,
            "turn": 22.0, # Head turned away while thinking
            "lag": 0.0
        },
        {
            "id": "clip_03_engaged_agreement",
            "desc": "High agreement, frequent head nodding, active vocal feedback",
            "duration": 4.0,
            "speech": [(0.3, 1.5), (1.9, 2.7), (3.0, 3.9)],
            "pitch": 210.0,
            "noise": False,
            "clip_audio": False,
            "lum": 140.0,
            "conf": 0.97,
            "nod": 2.2, # Fast nodding
            "turn": 3.0,
            "lag": 0.0
        },
        {
            "id": "clip_04_audio_clipping",
            "desc": "Sudden volume burst with distorted microphone clipping (Flagged)",
            "duration": 4.0,
            "speech": [(0.5, 3.5)],
            "pitch": 240.0,
            "noise": False,
            "clip_audio": True, # Hard clipping
            "lum": 130.0,
            "conf": 0.95,
            "nod": 0.0,
            "turn": 0.0,
            "lag": 0.0
        },
        {
            "id": "clip_05_dim_lighting",
            "desc": "Underexposed webcam video below luminance threshold (Flagged)",
            "duration": 4.0,
            "speech": [(0.6, 2.0), (2.5, 3.7)],
            "pitch": 185.0,
            "noise": False,
            "clip_audio": False,
            "lum": 18.0, # Severe dim lighting (<30)
            "conf": 0.72,
            "nod": 0.2,
            "turn": 4.0,
            "lag": 0.0
        },
        {
            "id": "clip_06_sync_drift",
            "desc": "Audio-video desynchronization, mouth moves 240ms before audio (Flagged)",
            "duration": 4.0,
            "speech": [(0.5, 2.0), (2.6, 3.8)],
            "pitch": 195.0,
            "noise": False,
            "clip_audio": False,
            "lum": 120.0,
            "conf": 0.95,
            "nod": 0.0,
            "turn": 2.0,
            "lag": 0.24 # 240ms lag (>120ms threshold)
        },
        {
            "id": "clip_07_side_profile_view",
            "desc": "Participant turned 35 degrees during deep listening (Gold Slice Edge Case)",
            "duration": 4.5,
            "speech": [(0.5, 1.4), (2.2, 4.0)],
            "pitch": 170.0,
            "noise": False,
            "clip_audio": False,
            "lum": 115.0,
            "conf": 0.88,
            "nod": 0.3,
            "turn": 35.0, # Large yaw
            "lag": 0.0
        },
        {
            "id": "clip_08_rapid_speech",
            "desc": "Fast-paced excited response with high speech density and rapid pauses",
            "duration": 3.5,
            "speech": [(0.2, 1.4), (1.6, 2.5), (2.7, 3.4)],
            "pitch": 225.0,
            "noise": False,
            "clip_audio": False,
            "lum": 130.0,
            "conf": 0.96,
            "nod": 1.1,
            "turn": 4.0,
            "lag": 0.0
        },
        {
            "id": "clip_09_subtle_micro_pause",
            "desc": "Deliberate 0.9s thinking pause with stable posture (Gold Slice Edge Case)",
            "duration": 4.5,
            "speech": [(0.5, 1.8), (2.7, 4.2)], # 0.9s pause
            "pitch": 180.0,
            "noise": False,
            "clip_audio": False,
            "lum": 125.0,
            "conf": 0.96,
            "nod": 0.1,
            "turn": 5.0,
            "lag": 0.0
        },
        {
            "id": "clip_10_noisy_cafe_background",
            "desc": "Conversation in loud ambient environment with poor SNR (Flagged)",
            "duration": 4.0,
            "speech": [(0.6, 2.0), (2.5, 3.6)],
            "pitch": 190.0,
            "noise": True,
            "noise_level": 0.12, # Severe noise
            "clip_audio": False,
            "lum": 120.0,
            "conf": 0.94,
            "nod": 0.3,
            "turn": 3.0,
            "lag": 0.0
        }
    ]

    manifest = []
    for cfg in clips_config:
        audio_path = os.path.join(output_dir, f"{cfg['id']}.wav")
        video_path = os.path.join(output_dir, f"{cfg['id']}_landmarks.json")
        
        generate_audio_wav(
            file_path=audio_path,
            duration_sec=cfg["duration"],
            pitch_hz=cfg["pitch"],
            speech_segments=cfg["speech"],
            add_noise=cfg["noise"],
            noise_level=cfg.get("noise_level", 0.02),
            clip_audio=cfg["clip_audio"]
        )
        
        generate_video_landmarks(
            file_path=video_path,
            duration_sec=cfg["duration"],
            speech_segments=cfg["speech"],
            mean_luminance=cfg["lum"],
            face_confidence=cfg["conf"],
            nod_frequency_hz=cfg["nod"],
            turn_away_deg=cfg["turn"],
            av_lag_sec=cfg["lag"]
        )

        manifest.append({
            "clip_id": cfg["id"],
            "description": cfg["desc"],
            "audio_file": audio_path,
            "video_file": video_path,
            "duration_sec": cfg["duration"]
        })

    # Automatically include any real downloaded clips present in output_dir
    import glob
    real_audios = sorted(glob.glob(os.path.join(output_dir, "real_*.wav")))
    for ra in real_audios:
        cid = os.path.splitext(os.path.basename(ra))[0]
        v_json = os.path.join(output_dir, f"{cid}_landmarks.json")
        if os.path.exists(v_json):
            manifest.append({
                "clip_id": cid,
                "description": f"Authentic real-world conversational recording ({cid})",
                "audio_file": ra,
                "video_file": v_json,
                "duration_sec": 6.0
            })

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
