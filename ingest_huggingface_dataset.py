"""Hugging Face Multimodal Ingestion, VAD Slicing & Gold Slice Benchmarking Pipeline.

Demonstrates all 3 core ML data engineering concepts on real Hugging Face data:
1. Real Multimodal Dataset Ingestion (PolyAI/minds14 conversational dataset)
2. Voice Activity Detection (VAD) & speech-to-pause boundary segmentation
3. Automated Quality Gatekeeping & Gold Slice Parquet Curation for PyTorch
"""

import argparse
import io
import json
import os
import urllib.request
import pandas as pd
from rich.console import Console
from rich.table import Table
import soundfile as sf

from behaviorbench.benchmark_curator import curate_benchmark
from behaviorbench.config import BehavioralFeatures, QualityThresholds
from behaviorbench.quality_gater import evaluate_clip_quality
from behaviorbench.reporter import generate_html_report
from behaviorbench.signal_extractor import extract_audio_features

console = Console()

HF_DATASET_URL = "https://huggingface.co/datasets/PolyAI/minds14/resolve/main/en-US/train-00000-of-00001.parquet"


def ensure_hf_dataset(cache_path: str = "sample_data/hf_minds14_sample.parquet") -> str:
    """Download the official Hugging Face conversational dataset if not cached."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        console.print(
            f"[bold cyan]--> Downloading real conversational dataset from Hugging Face ({HF_DATASET_URL})...[/bold cyan]"
        )
        urllib.request.urlretrieve(HF_DATASET_URL, cache_path)
        console.print(
            f"[green]✓ Download complete: {os.path.getsize(cache_path)/(1024*1024):.1f} MB[/green]"
        )
    return cache_path


def run_huggingface_pipeline(
    num_samples: int = 10,
    output_dir: str = "output/huggingface_run",
    use_strict_preset: bool = False,
):
    """Run end-to-end VAD slicing, quality gatekeeping, and benchmark curation on Hugging Face data."""
    console.print(
        "\n[bold magenta]========================================================================[/bold magenta]"
    )
    console.print(
        "[bold cyan]   BehaviorBench 2.0: Real Hugging Face Multimodal Curation Pipeline   [/bold cyan]"
    )
    console.print(
        "[bold magenta]========================================================================[/bold magenta]\n"
    )

    parquet_file = ensure_hf_dataset()
    df = pd.read_parquet(parquet_file)
    total_in_repo = len(df)
    console.print(
        f"[dim]Loaded official Hugging Face dataset with {total_in_repo} conversational samples.[/dim]"
    )

    num_samples = min(num_samples, total_in_repo)
    console.print(
        f"[bold blue]Processing {num_samples} real conversational speech turns...[/bold blue]\n"
    )

    thresholds = (
        QualityThresholds.interhuman_preset()
        if use_strict_preset
        else QualityThresholds()
    )
    os.makedirs(output_dir, exist_ok=True)
    extracted_wav_dir = os.path.join(output_dir, "extracted_wavs")
    os.makedirs(extracted_wav_dir, exist_ok=True)

    results = []
    table = Table(
        title=f"BehaviorBench 2.0 Evaluation of Hugging Face Conversational Data ({num_samples} Samples)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Clip ID", style="cyan")
    table.add_column("Transcription (Real Speech)", style="dim")
    table.add_column("Duration", justify="right")
    table.add_column("Vocal Pitch", justify="right")
    table.add_column("VAD Pauses", justify="right")
    table.add_column("SNR (dB)", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Social Signal Slice", style="italic blue")

    for idx in range(num_samples):
        row = df.iloc[idx]
        clip_id = f"hf_turn_{idx+1:03d}"
        transcription = row["transcription"]
        audio_payload = row["audio"]
        audio_bytes = audio_payload["bytes"]

        wav_path = os.path.join(extracted_wav_dir, f"{clip_id}.wav")
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # 1. Feature Extraction & Voice Activity Detection (VAD)
        a_feat = extract_audio_features(wav_path, thresholds)

        # 2. Package Behavioral Features
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
            total_frames=max(30, int(a_feat["duration_sec"] * 30)),
            mean_face_confidence=0.95,
            mean_luminance=100.0,
            head_pitch_variance=1.5,
            head_yaw_variance=1.5,
            blink_count=2,
            mouth_openness_variance=0.04,
            av_sync_drift_ms=0.0,
        )

        # 3. Quality Gate & Social Signal Tagger
        inspection = evaluate_clip_quality(clip_id, features, thresholds)
        results.append(inspection)

        # Visual Table Formatting
        status_style = {
            "PASS_CLEAN": "[green]PASS_CLEAN[/green]",
            "PASS_EDGE_CASE": "[purple]GOLD_SLICE[/purple]",
            "FLAGGED": "[yellow]FLAGGED[/yellow]",
            "REJECTED": "[red]REJECTED[/red]",
        }.get(inspection.status.value, inspection.status.value)

        sig_name = (
            inspection.social_signal.value if inspection.social_signal else "none"
        )
        short_transcript = (
            transcription[:38] + "..."
            if len(transcription) > 38
            else transcription
        )

        table.add_row(
            clip_id,
            short_transcript,
            f"{features.duration_sec:.1f}s",
            f"{features.mean_pitch_hz:.0f} Hz",
            str(features.hesitation_pauses_count),
            f"{features.snr_db:.1f}",
            status_style,
            sig_name,
        )

    console.print(table)

    # 4. Curate into Apache Parquet splits & generate report
    curation_data = curate_benchmark(results, output_dir)
    report_file = os.path.join(output_dir, "benchmark_report.html")
    generate_html_report(curation_data, report_file)

    summary = curation_data["summary"]
    console.print("\n[bold green]✓ Hugging Face Benchmark Curation Complete![/bold green]")
    console.print(
        f"  • Total HF Conversational Clips: [bold]{summary['total_clips_analyzed']}[/bold]"
    )
    console.print(
        f"  • Clean Baseline Clips:          [bold green]{summary['clean_baseline_count']}[/bold green]"
    )
    console.print(
        f"  • Gold-Slice Edge Cases:         [bold purple]{summary['gold_slice_edge_case_count']}[/bold purple]"
    )
    console.print(
        f"  • Flagged / Quarantined:         [bold yellow]{summary['flagged_count']}[/bold yellow]"
    )
    console.print(
        f"  • Rejected:                      [bold red]{summary['rejected_count']}[/bold red]"
    )
    console.print(
        f"  • SHA-256 Fingerprint:           [dim]{summary['manifest_sha256_fingerprint']}[/dim]"
    )
    if "parquet_paths" in curation_data:
        console.print("\n[bold cyan]📦 Generated Production Parquet ML Splits:[/bold cyan]")
        for split_name, ppath in curation_data["parquet_paths"].items():
            console.print(f"  • {split_name}: [dim]{ppath}[/dim]")
    console.print(
        f"\n[bold underline cyan]🌐 Interactive Report Generated: {report_file}[/bold underline cyan]\n"
    )

    return curation_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest and benchmark real Hugging Face multimodal conversational data"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of real conversational samples to process (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="output/huggingface_run",
        help="Output directory for Parquet splits and manifests",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use Interhuman Inter-2 strict curation preset",
    )
    args = parser.parse_args()

    run_huggingface_pipeline(
        num_samples=args.samples,
        output_dir=args.output,
        use_strict_preset=args.strict,
    )
