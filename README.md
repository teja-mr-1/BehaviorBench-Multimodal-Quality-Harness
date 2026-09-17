# BehaviorBench 🎯

**Automated Data Quality Gatekeeper & "Gold Slice" Benchmark Harness for Multimodal Behavioral AI**

![BehaviorBench Dashboard](output/dashboard_screenshot.png)

---

## 📌 Problem Overview

Building AI systems that interpret non-verbal human communication (hesitation, vocal tone, engagement, facial expressions) is bottlenecked by **real-world input noise**:
* **Audio Clipping:** Microphone distortion and volume bursts corrupt vocal frequency analysis.
* **Ambient Noise & Echo:** Room reverberation and low SNR degrade acoustic prosody extraction.
* **Lighting & Occlusion:** Underexposed video or sudden face occlusions break landmark tracking.
* **Audio-Video Desynchronization:** Timestamp drift between mouth movement and vocal onset causes multi-modal model confusion.
* **Edge-Case Annotation Gaps:** Subtle human cues (e.g. reflective pauses, nodding in agreement, looking away while thinking) get misclassified unless specifically tracked in benchmark splits.

**BehaviorBench** is an automated data quality pipeline, web application, and benchmark curation suite designed to ingest raw conversational video and audio streams, evaluate signal integrity against strict quality gates, and isolate high-value **"Gold Slice"** edge cases for regression testing.

---

## 🏗️ Architecture & Pipeline Flow

```
Raw Media Ingestion (Video MP4 / Audio WAV / Real-Time Upload)
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Multimodal Signal Extraction                 │
│  - Acoustic: Frame RMS, VAD, Pause Duration, Pitch     │
│  - Visual: Video Frame Luminance, Pose Stability       │
│  - Temporal: Cross-Correlation AV Sync Drift (ms)      │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Automated Quality Gatekeeping                │
│  - Clipping Gate: Threshold <= 1.5% max amplitude      │
│  - SNR Gate: Minimum >= 12.0 dB                        │
│  - Lighting Gate: Luminance >= 30 (0-255 scale)        │
│  - AV Sync Drift Gate: Absolute drift <= 120 ms        │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: "Gold Slice" Benchmark Curation              │
│  - Clean Baseline: Standard conversational reference   │
│  - Gold-Slice Edge Cases: Subtle hesitations & turns   │
│  - Cryptographic Fingerprint: SHA-256 Dataset Hash     │
│  - Manifest Exports: JSON & CSV                        │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│  Phase 4: Multi-Channel Interface                      │
│  - Streamlit Cloud Web Application (Live File Upload)  │
│  - Standalone Responsive HTML Dashboard Report         │
│  - Rich Terminal CLI Inspection Table                  │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Navigate to project
cd /mnt/linuxdata/Projects/behavior-bench

# 2. Activate virtual environment or install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch the Interactive Web Application (Upload video/audio live)
streamlit run app.py

# 4. Or run the automated benchmark runner with real & synthetic clips
python run_demo.py
```

---

## 🌐 Live Cloud Deployment (Free & Immediate)

You can deploy this application publicly so users and recruiters can drag-and-drop video/audio files directly in their browser:

### Option 1: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub: `https://github.com/teja-mr-1/BehaviorBench-Multimodal-Quality-Harness`
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Select your repository, branch `main`, and set Main file path to `app.py`.
4. The application automatically detects `packages.txt` (`ffmpeg`) and `requirements.txt`.
5. Deploy! Your app will be live at `https://behaviorbench.streamlit.app`.
6. Embed it directly in your portfolio card via iframe using `?embed=true`.

### Option 2: Hugging Face Spaces
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Select **Streamlit** SDK.
3. Push files to the Hugging Face Space Git remote.
4. Live instance provides 16 GB RAM and free public hosting.

---

## 📊 Benchmark Curation Logic

| Category | Criteria | Purpose |
| :--- | :--- | :--- |
| **Clean Baseline** | Passed all gates, normal cadence, steady posture | Ensures models maintain baseline accuracy on standard conversational speech. |
| **Gold Slice (Edge Cases)** | Passed all gates, but contains long pauses (>0.8s), expressive head nods, or lateral head turns (>18°) | Tests model robustness against subtle, non-verbal behavioral nuances without corrupted data. |
| **Flagged** | Non-critical warnings (e.g. ambient room echo, low SNR) | Quarantined for manual review before entering training splits. |
| **Rejected** | Hard failures (clipping >1.5%, severe darkness, sync drift >120ms) | Blocked from ever entering model training runs or evaluations. |

---

## 🛠️ Project Structure

```
/mnt/linuxdata/Projects/behavior-bench/
├── app.py                     # Streamlit web application (drag-and-drop video/audio upload)
├── packages.txt               # System dependencies (ffmpeg) for Streamlit Cloud
├── requirements.txt           # Python dependencies (Streamlit, Plotly, SciPy, Pydantic)
├── run_demo.py                # One-command CLI & batch runner
├── ingest_real_video.py       # FFmpeg video ingestion & frame sampling script
├── behaviorbench/
│   ├── __init__.py
│   ├── config.py              # Pydantic schemas & quality thresholds
│   ├── data_generator.py      # Conversational audio-video synthesizer
│   ├── signal_extractor.py    # Acoustic, visual & AV sync feature extractor
│   ├── quality_gater.py       # Automated pass/flag/reject gatekeeper
│   ├── benchmark_curator.py   # Gold-slice partitioner & SHA-256 hasher
│   ├── reporter.py            # Responsive HTML dashboard generator
│   └── cli.py                 # Rich CLI interface
├── sample_data/               # Real interview clips (.mp4) and test media (.wav, .json)
└── output/
    ├── benchmark_manifest.json# Complete auditable JSON benchmark
    ├── benchmark_summary.csv  # Flat summary table
    └── benchmark_report.html  # Interactive HTML dashboard
```

---

## 🎯 Startup Alignment

Built to mirror daily data engineering in applied behavioral AI teams (such as **Interhuman AI**):
1. **Lightweight & Pragmatic:** Zero heavy framework bloat; runs in seconds on CPU.
2. **Automates Unsexy Work:** Catches corrupted audio and video streams before wasting expensive GPU training compute.
3. **100% Repeatable:** Deterministic SHA-256 fingerprinting ensures regression benchmarks stay locked across model releases.

---

**Author:** Raviteja Maddela &bull; MSc Computer Science & Engineering, Technical University of Denmark (DTU)
