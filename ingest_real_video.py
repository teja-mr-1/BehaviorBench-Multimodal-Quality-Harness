"""Helper script to ingest and analyze any REAL .mp4 video file with BehaviorBench."""

import argparse
import json
import os
import subprocess
from behaviorbench.config import BehavioralFeatures, QualityThresholds
from behaviorbench.quality_gater import evaluate_clip_quality
from behaviorbench.signal_extractor import (
    estimate_av_sync_drift,
    extract_audio_features,
    extract_video_features,
)
from behaviorbench.vision_tracker import VideoVisionTracker


def ingest_real_mp4(mp4_path: str, output_dir: str = "./sample_data"):
    """Extract audio and video frame metadata from a real MP4 video using OpenCV and ffmpeg."""
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Video file not found: {mp4_path}")

    os.makedirs(output_dir, exist_ok=True)
    clip_id = os.path.splitext(os.path.basename(mp4_path))[0]

    audio_path = os.path.join(output_dir, f"{clip_id}.wav")
    landmarks_path = os.path.join(output_dir, f"{clip_id}_landmarks.json")

    print(f"--> Extracting 16kHz audio from {mp4_path} via ffmpeg...")
    cmd_audio = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    subprocess.run(
        cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    print(f"--> Tracking real facial dynamics, mouth motion, and pose via OpenCV...")
    tracker = VideoVisionTracker()
    vision_results = tracker.track_video(mp4_path, target_fps=30)

    # Save landmarks JSON for downstream benchmark manifests
    with open(landmarks_path, "w") as f:
        json.dump(
            {
                "fps": vision_results["fps"],
                "total_frames": vision_results["total_frames"],
                "duration_sec": vision_results["duration_sec"],
                "mean_face_confidence": vision_results["mean_face_confidence"],
                "face_detection_ratio": vision_results["face_detection_ratio"],
                "mean_luminance": vision_results["mean_luminance"],
                "head_pitch_variance": vision_results["head_pitch_variance"],
                "head_yaw_variance": vision_results["head_yaw_variance"],
                "mouth_openness_variance": vision_results["mouth_openness_variance"],
                "frames": vision_results["frames"],
            },
            f,
            indent=2,
        )

    print(f"✓ Ingested real clip: {clip_id}")
    print(f"  • Audio: {audio_path}")
    print(
        f"  • Video Metadata: {landmarks_path} ({vision_results['total_frames']} frames, {vision_results['duration_sec']:.1f}s)"
    )
    print(f"  • Face Detection Ratio: {vision_results['face_detection_ratio']*100:.1f}%")
    print(f"  • Real Mouth Variance: {vision_results['mouth_openness_variance']:.4f}")
    return clip_id, audio_path, landmarks_path, vision_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a real .mp4 video into BehaviorBench"
    )
    parser.add_argument("video_path", help="Path to real .mp4 video file")
    parser.add_argument("--output", default="./sample_data", help="Output directory")
    args = parser.parse_args()

    clip_id, a_path, v_path, v_res = ingest_real_mp4(args.video_path, args.output)

    # Run immediate quality evaluation
    thresholds = QualityThresholds()
    a_feat = extract_audio_features(a_path, thresholds)
    v_feat = extract_video_features(v_path)

    # Estimate real AV sync drift
    sync_drift = estimate_av_sync_drift(
        audio_rms=a_feat.get("rms_envelope"),
        audio_hop_len=a_feat.get("hop_len", 400),
        audio_rate=a_feat.get("sample_rate", 16000),
        mouth_series=v_res.get("mouth_series"),
        video_fps=int(v_res.get("fps", 30)),
    )

    features = BehavioralFeatures(
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
        av_sync_drift_ms=sync_drift,
    )
    result = evaluate_clip_quality(clip_id, features, thresholds)
    print("\n--- Real Video Quality Gate Result ---")
    print(f"Status: {result.status.value}")
    print(f"Benchmark Slice: {result.benchmark_slice}")
    print(f"Quality Score: {result.quality_score} / 100")
    print(f"Mean Pitch: {features.mean_pitch_hz} Hz")
    print(f"AV Sync Drift: {features.av_sync_drift_ms} ms")
    print(f"Silence Ratio: {features.silence_ratio*100:.1f}%")
    print(
        f"Anomalies: {result.anomalies_detected if result.anomalies_detected else 'None (Clean)'}"
    )
