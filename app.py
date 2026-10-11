"""BehaviorBench 2.0: Interactive Multimodal Quality Gatekeeper & Benchmark Web Application.

Aligned with Interhuman.ai Inter-2 social signal research and multimodal data curation.
Run locally:
    streamlit run app.py
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import soundfile as sf
import streamlit as st

from behaviorbench.benchmark_curator import curate_benchmark
from behaviorbench.config import (
    BehavioralFeatures,
    ClipInspectionResult,
    ClipStatus,
    QualityThresholds,
    SocialSignalCategory,
)
from behaviorbench.quality_gater import evaluate_clip_quality
from behaviorbench.signal_extractor import (
    estimate_av_sync_drift,
    extract_audio_features,
    extract_video_features,
)
from behaviorbench.vision_tracker import VideoVisionTracker

# Page configuration
st.set_page_config(
    page_title="BehaviorBench 2.0 | Multimodal Quality Gatekeeper",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0284c7;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
    .status-pass-clean { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .status-pass-edge { background: #f3e8ff; color: #6b21a8; border: 1px solid #d8b4fe; }
    .status-flagged { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .status-rejected { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .interhuman-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def extract_media(file_path: str, is_video: bool):
    """Decompose audio and extract computer vision features using OpenCV and ffmpeg."""
    temp_dir = tempfile.mkdtemp()
    audio_wav = os.path.join(temp_dir, "audio_16k.wav")

    # Extract 16kHz mono audio via ffmpeg
    cmd_audio = [
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_wav,
    ]
    subprocess.run(
        cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    frames_metadata = []
    mouth_series = np.zeros(1)
    if is_video:
        try:
            tracker = VideoVisionTracker()
            v_res = tracker.track_video(file_path, target_fps=30)
            frames_metadata = v_res["frames"]
            mouth_series = v_res["mouth_series"]
        except Exception as e:
            st.warning(f"Video CV tracking encountered a warning: {e}. Falling back to default frame sampling.")

    return audio_wav, frames_metadata, mouth_series, temp_dir


# Sidebar Controls
st.sidebar.markdown("### ⚙️ Curation Profiles & Thresholds")

profile_choice = st.sidebar.selectbox(
    "Target Quality Profile:",
    [
        "Interhuman Inter-2 Preset (Strict Curation)",
        "Zero-Hallucination Strict (Phantom Audio Gate)",
        "Standard Balanced Ingestion",
        "Custom Thresholds",
    ],
    index=0,
)

if profile_choice == "Interhuman Inter-2 Preset (Strict Curation)":
    default_snr = 14.0
    default_clipping = 0.8
    default_lum = 35.0
    default_sync = 60.0
    default_pause = 0.65
elif profile_choice == "Zero-Hallucination Strict (Phantom Audio Gate)":
    default_snr = 16.0
    default_clipping = 0.5
    default_lum = 40.0
    default_sync = 45.0
    default_pause = 0.50
elif profile_choice == "Standard Balanced Ingestion":
    default_snr = 12.0
    default_clipping = 1.5
    default_lum = 30.0
    default_sync = 120.0
    default_pause = 0.80
else:
    default_snr = 12.0
    default_clipping = 1.5
    default_lum = 30.0
    default_sync = 120.0
    default_pause = 0.80

threshold_snr = st.sidebar.slider(
    "Min Audio SNR (dB)",
    min_value=5.0,
    max_value=25.0,
    value=float(default_snr),
    step=0.5,
)
threshold_clipping = st.sidebar.slider(
    "Max Clipping Ratio (%)",
    min_value=0.2,
    max_value=5.0,
    value=float(default_clipping),
    step=0.1,
)
threshold_lum = st.sidebar.slider(
    "Min Luminance (0-255)",
    min_value=15.0,
    max_value=60.0,
    value=float(default_lum),
    step=1.0,
)
threshold_sync = st.sidebar.slider(
    "Max AV Sync Drift (ms)",
    min_value=30.0,
    max_value=300.0,
    value=float(default_sync),
    step=5.0,
)
threshold_pause = st.sidebar.slider(
    "Hesitation Pause Threshold (s)",
    min_value=0.3,
    max_value=2.0,
    value=float(default_pause),
    step=0.05,
)

custom_thresholds = QualityThresholds(
    min_snr_db=threshold_snr,
    max_clipping_ratio=threshold_clipping / 100.0,
    min_luminance=threshold_lum,
    max_av_drift_ms=threshold_sync,
    strict_av_drift_ms=min(45.0, threshold_sync / 2.0),
    hesitation_pause_threshold_sec=threshold_pause,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About BehaviorBench 2.0")
st.sidebar.markdown(
    """
**BehaviorBench 2.0** is an enterprise-grade multimodal data quality gatekeeper and "Gold Slice" benchmark harness.
Specifically tuned for conversational AI pipelines and omni-modal foundation models (**Interhuman Inter-2**).

* **Author:** Raviteja Maddela (DTU MSc Computer Science)
* **Codebase:** [GitHub Repository](https://github.com/teja-mr-1/BehaviorBench-Multimodal-Quality-Harness)
* **Stack:** OpenCV, Librosa, SciPy, PyArrow, Streamlit
"""
)

# Main Dashboard Header
st.markdown(
    '<div class="main-header">BehaviorBench 2.0 🎯</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Automated Multimodal Data Quality Gatekeeper & "Gold Slice" Benchmark Suite for Behavioral AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="interhuman-banner">
    <strong>⚡ Production Focus:</strong> Automated quality gatekeeping for multimodal conversational datasets (Zoom, interviews, in-the-wild video). 
    Filters out ruined training data (audio clipping, ambient noise, packet sync drift) and categorizes high-value <strong>Gold Slice Edge Cases</strong> (Hesitation, Nodding, Cognitive Turns) directly exported to <strong>Apache Parquet</strong>.
</div>
""",
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_inspect, tab_batch, tab_about = st.tabs(
    ["🔍 Live Clip Inspection", "📊 Benchmark Batch Runner & Parquet Hub", "📖 Architecture & Interhuman Mapping"]
)

with tab_inspect:
    st.markdown("### 1. Upload Conversational Video or Audio Stream")
    uploaded_file = st.file_uploader(
        "Choose an MP4 video or WAV audio file for automated gatekeeping:",
        type=["mp4", "wav", "mov", "avi"],
    )

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    load_sample = None
    with sample_col1:
        if st.button("Load Real Researcher Video (MP4)"):
            load_sample = "sample_data/real_interview_male_researcher.mp4"
    with sample_col2:
        if st.button("Load Thoughtful Hesitation Audio (WAV)"):
            load_sample = "sample_data/clip_02_thoughtful_hesitant.wav"
    with sample_col3:
        if st.button("Load Clipped Audio Stream (WAV)"):
            load_sample = "sample_data/clip_04_audio_clipping.wav"

    target_path = None
    is_video = False
    temp_work_dir = None

    if uploaded_file is not None:
        temp_work_dir = tempfile.mkdtemp()
        target_path = os.path.join(temp_work_dir, uploaded_file.name)
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        is_video = uploaded_file.name.lower().endswith((".mp4", ".mov", ".avi"))
        clip_id = os.path.splitext(uploaded_file.name)[0]
    elif load_sample is not None and os.path.exists(load_sample):
        target_path = load_sample
        is_video = load_sample.lower().endswith((".mp4", ".mov", ".avi"))
        clip_id = os.path.splitext(os.path.basename(load_sample))[0]

    if target_path:
        st.markdown("---")
        with st.spinner("Extracting multimodal acoustic & visual features via OpenCV and Librosa..."):
            audio_wav, frames_meta, mouth_arr, cleanup_dir = extract_media(
                target_path, is_video=is_video
            )

            a_feat = extract_audio_features(audio_wav, custom_thresholds)
            v_feat = extract_video_features(frames_meta)

            sync_drift = estimate_av_sync_drift(
                audio_rms=a_feat["rms_envelope"],
                audio_hop_len=a_feat["hop_len"],
                audio_rate=a_feat["sample_rate"],
                mouth_series=mouth_arr if len(mouth_arr) > 1 else v_feat["mouth_series"],
                video_fps=int(v_feat.get("fps", 30)),
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
                zero_crossing_rate=a_feat.get("zero_crossing_rate", 0.05),
                total_frames=v_feat["total_frames"],
                mean_face_confidence=v_feat["mean_face_confidence"],
                mean_luminance=v_feat["mean_luminance"],
                head_pitch_variance=v_feat["head_pitch_variance"],
                head_yaw_variance=v_feat["head_yaw_variance"],
                blink_count=v_feat["blink_count"],
                mouth_openness_variance=v_feat["mouth_openness_variance"],
                av_sync_drift_ms=sync_drift,
            )

            res = evaluate_clip_quality(clip_id, features, custom_thresholds)

        # Gatekeeper Verdict Header
        status_map = {
            ClipStatus.PASS_CLEAN: (
                "PASS (CLEAN BASELINE)",
                "status-pass-clean",
                "Ready for foundation model training baseline.",
            ),
            ClipStatus.PASS_EDGE_CASE: (
                "PASS (GOLD SLICE EDGE CASE)",
                "status-pass-edge",
                "High-value conversational edge case preserved for regression testing.",
            ),
            ClipStatus.FLAGGED: (
                "FLAGGED (QUARANTINED)",
                "status-flagged",
                "Minor signal degradation detected. Quarantined for manual audit.",
            ),
            ClipStatus.REJECTED: (
                "REJECTED (CONTAMINATED)",
                "status-rejected",
                "Hard quality failure. Blocked from training pipelines.",
            ),
        }
        title, badge_css, explanation = status_map[res.status]

        st.markdown(
            f"""
        <div style="background: #ffffff; padding: 22px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span class="status-badge {badge_css}">{title}</span>
                    <div style="font-size: 0.95rem; color: #475569; margin-top: 8px;">{explanation}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{res.quality_score} <span style="font-size: 1.1rem; color: #64748b;">/ 100</span></div>
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700;">Data Integrity Score</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Behavioral & Social Signal Slice
        sig_label = res.social_signal.value if res.social_signal else "None"
        slice_label = res.benchmark_slice or "None (Excluded)"
        st.info(
            f"🎯 **Targeted Social Signal Slice:** `{sig_label}` | **Benchmark Partition:** `{slice_label}`"
        )

        if res.anomalies_detected:
            st.error("⚠️ **Quality Violations Flagged:**\n* " + "\n* ".join(res.anomalies_detected))
        else:
            st.success("✅ **Zero Quality Violations:** Signal strictly complies with selected curation gate.")

        # Key Metrics Grid
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        with m_c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Audio Clipping</div><div class="metric-value">{features.clipping_ratio*100:.2f}%</div></div>',
                unsafe_allow_html=True,
            )
        with m_c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Signal-to-Noise</div><div class="metric-value">{features.snr_db:.1f} dB</div></div>',
                unsafe_allow_html=True,
            )
        with m_c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">AV Sync Drift</div><div class="metric-value">{features.av_sync_drift_ms:+.1f} ms</div></div>',
                unsafe_allow_html=True,
            )
        with m_c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Hesitation Pauses</div><div class="metric-value">{features.hesitation_pauses_count}</div></div>',
                unsafe_allow_html=True,
            )
        with m_c5:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Mean Vocal Pitch</div><div class="metric-value">{features.mean_pitch_hz:.0f} Hz</div></div>',
                unsafe_allow_html=True,
            )
        with m_c6:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Face Confidence</div><div class="metric-value">{features.mean_face_confidence*100:.0f}%</div></div>',
                unsafe_allow_html=True,
            )

        # Multi-Track Synchronized Plotly Timeline
        st.markdown("### 2. Multi-Signal Synchronized Time-Series Inspection")

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Track 1: Acoustic Energy (RMS Envelope) & Hesitation Pauses",
                "Track 2: Real Facial Dynamics & Mouth Kinetic Velocity (OpenCV)",
                "Track 3: Cross-Correlation Lag & Synchronization Window",
            ),
        )

        # Track 1: Audio RMS envelope
        t_audio = np.linspace(0, features.duration_sec, len(a_feat["rms_envelope"]))
        fig.add_trace(
            go.Scatter(
                x=t_audio,
                y=a_feat["rms_envelope"],
                mode="lines",
                name="Audio RMS",
                line=dict(color="#0284c7", width=2),
            ),
            row=1,
            col=1,
        )

        # Highlight pause intervals in Track 1
        for s_start, s_end in a_feat.get("speech_intervals", []):
            pass  # speech active

        # Track 2: Visual mouth dynamics
        if len(mouth_arr) > 1:
            t_video = np.linspace(0, features.duration_sec, len(mouth_arr))
            fig.add_trace(
                go.Scatter(
                    x=t_video,
                    y=mouth_arr,
                    mode="lines",
                    name="Mouth Kinetic Score",
                    line=dict(color="#7c3aed", width=2),
                ),
                row=2,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0, features.duration_sec],
                    y=[0.05, 0.05],
                    mode="lines",
                    name="Visual Stream",
                    line=dict(color="#94a3b8", dash="dash"),
                ),
                row=2,
                col=1,
            )

        # Track 3: Drift Indicator
        fig.add_trace(
            go.Scatter(
                x=[0, features.duration_sec],
                y=[features.av_sync_drift_ms, features.av_sync_drift_ms],
                mode="lines",
                name="Estimated AV Drift (ms)",
                line=dict(color="#e11d48", width=2, dash="dashdot"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=600,
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
        )
        fig.update_xaxes(title_text="Timeline (seconds)", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Media Player Playback
        if is_video and os.path.exists(target_path):
            st.video(target_path)
        elif os.path.exists(audio_wav):
            st.audio(audio_wav)

        # Cleanup temp directory
        if temp_work_dir and os.path.exists(temp_work_dir):
            shutil.rmtree(temp_work_dir, ignore_errors=True)
        if cleanup_dir and os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


with tab_batch:
    st.markdown("### Batch Curation & ML Training Split Hub")
    st.markdown(
        "Run the automated gatekeeper across all conversational video/audio clips in the dataset, generate deterministic SHA-256 fingerprints, and export **Apache Parquet (`.parquet`)** splits for PyTorch."
    )

    sample_dir = os.path.abspath("sample_data")
    output_dir = os.path.abspath("output")

    if st.button("🚀 Execute Batch Benchmark & Generate Parquet Splits"):
        with st.spinner("Executing pipeline and compiling benchmark splits..."):
            manifest_file = os.path.join(sample_dir, "dataset_manifest.json")
            if os.path.exists(manifest_file):
                with open(manifest_file, "r") as f:
                    clips = json.load(f)
            else:
                clips = []

            batch_results: List[ClipInspectionResult] = []
            for c in clips:
                cid = c["clip_id"]
                a_file = os.path.join(sample_dir, os.path.basename(c["audio_file"]))
                v_file = os.path.join(sample_dir, os.path.basename(c["video_file"]))
                if os.path.exists(a_file) and os.path.exists(v_file):
                    feats = extract_multimodal_features(
                        a_file, v_file, thresholds=custom_thresholds
                    )
                    r = evaluate_clip_quality(cid, feats, custom_thresholds)
                    batch_results.append(r)

            curation_out = curate_benchmark(batch_results, output_dir)
            st.success(
                f"✅ Curated {len(batch_results)} clips! SHA-256 Fingerprint: `{curation_out['summary']['manifest_sha256_fingerprint']}`"
            )

    # Check existing outputs
    summary_csv = os.path.join(output_dir, "benchmark_summary.csv")
    gold_parquet = os.path.join(output_dir, "benchmark_gold_slice_edgecases.parquet")
    manifest_json = os.path.join(output_dir, "benchmark_manifest.json")
    report_html = os.path.join(output_dir, "benchmark_report.html")

    if os.path.exists(summary_csv):
        df_summary = pd.read_csv(summary_csv)
        st.dataframe(
            df_summary[
                [
                    "clip_id",
                    "status",
                    "benchmark_slice",
                    "social_signal",
                    "quality_score",
                    "hesitation_pauses",
                    "snr_db",
                    "av_sync_drift_ms",
                    "anomalies",
                ]
            ],
            use_container_width=True,
        )

        down_col1, down_col2, down_col3, down_col4 = st.columns(4)
        with down_col1:
            if os.path.exists(gold_parquet):
                with open(gold_parquet, "rb") as f:
                    st.download_button(
                        "📦 Download Gold Slice (Parquet)",
                        data=f.read(),
                        file_name="benchmark_gold_slice_edgecases.parquet",
                        mime="application/octet-stream",
                    )
        with down_col2:
            if os.path.exists(manifest_json):
                with open(manifest_json, "r") as f:
                    st.download_button(
                        "📄 Download Manifest (JSON)",
                        data=f.read(),
                        file_name="benchmark_manifest.json",
                        mime="application/json",
                    )
        with down_col3:
            with open(summary_csv, "rb") as f:
                st.download_button(
                    "📊 Download Summary (CSV)",
                    data=f.read(),
                    file_name="benchmark_summary.csv",
                    mime="text/csv",
                )
        with down_col4:
            if os.path.exists(report_html):
                with open(report_html, "rb") as f:
                    st.download_button(
                        "🌐 Download Audit Report (HTML)",
                        data=f.read(),
                        file_name="benchmark_report.html",
                        mime="text/html",
                    )


with tab_about:
    st.markdown("### 🎯 Architecture Alignment with Interhuman.ai")
    st.markdown(
        """
#### 1. Why BehaviorBench Exists
Multimodal behavioral AI (such as Interhuman's **Inter-2** omni-modal model) processes text, vocal tone, and facial dynamics.
However, training on raw, unfiltered in-the-wild video creates fatal failure modes:
* **Audio Clipping:** Harsh digital saturation distorts voice pitch (F0) tracking.
* **Low SNR / Ambient Noise:** Distorts whisper and hesitation detection.
* **Timestamp Misalignment (AV Sync Drift):** Audio lagging behind facial landmarks trains foundation models on fake cross-modal associations (e.g. hallucinated utterances).

#### 2. The Solution: BehaviorBench 2.0
BehaviorBench acts as an automated quality gatekeeper and benchmark curator:
1. **Gatekeeping:** Purges bad data before expensive GPU training compute is consumed.
2. **Gold Slice Curation:** Preserves high-value conversational nuances (Hesitations, Nodding, Cognitive Turns) in standardized test splits.
3. **ML Infrastructure Integration:** Automatically partitions data into **Apache Parquet (`.parquet`)** formats ready for PyTorch `DataLoaders`.
"""
    )
