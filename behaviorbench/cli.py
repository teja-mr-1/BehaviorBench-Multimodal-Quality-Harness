"""Command-line interface for BehaviorBench 2.0."""

import argparse
import glob
import json
import os
import sys
from rich.console import Console
from rich.table import Table

from behaviorbench.benchmark_curator import curate_benchmark
from behaviorbench.config import QualityThresholds
from behaviorbench.data_generator import generate_synthetic_dataset
from behaviorbench.quality_gater import evaluate_clip_quality
from behaviorbench.reporter import generate_html_report
from behaviorbench.signal_extractor import extract_multimodal_features

console = Console()


def run_pipeline(data_dir: str, output_dir: str, use_interhuman_preset: bool = False):
    """Run full multimodal inspection, quality gating, and benchmark curation."""
    console.print(
        "[bold blue]Starting BehaviorBench 2.0 Multimodal Quality & Benchmark Pipeline...[/bold blue]"
    )
    if use_interhuman_preset:
        console.print("[dim cyan]Mode: Using Interhuman Inter-2 Strict Curation Preset[/dim cyan]")
        thresholds = QualityThresholds.interhuman_preset()
    else:
        thresholds = QualityThresholds()

    manifest_file = os.path.join(data_dir, "dataset_manifest.json")
    if not os.path.exists(manifest_file):
        console.print(
            f"[yellow]Manifest not found at {manifest_file}. Scanning directory for WAV files...[/yellow]"
        )
        wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
        clips = []
        for w in wav_files:
            cid = os.path.splitext(os.path.basename(w))[0]
            v_json = os.path.join(data_dir, f"{cid}_landmarks.json")
            if os.path.exists(v_json):
                clips.append({"clip_id": cid, "audio_file": w, "video_file": v_json})
    else:
        with open(manifest_file, "r") as f:
            clips = json.load(f)

    if not clips:
        console.print("[bold red]No valid multimodal clips found to inspect![/bold red]")
        sys.exit(1)

    results = []

    table = Table(
        title="BehaviorBench 2.0 Multimodal Quality Gate & Social Signal Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Clip ID", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Social Signal Slice", style="italic blue")
    table.add_column("Audio Clip %", justify="right")
    table.add_column("Pauses", justify="right")
    table.add_column("AV Drift (ms)", justify="right")
    table.add_column("Anomalies Flagged")

    for clip in clips:
        cid = clip["clip_id"]
        features = extract_multimodal_features(
            wav_path=clip["audio_file"],
            landmarks_path=clip["video_file"],
            thresholds=thresholds,
        )
        res = evaluate_clip_quality(cid, features, thresholds)
        results.append(res)

        status_style = {
            "PASS_CLEAN": "[green]PASS_CLEAN[/green]",
            "PASS_EDGE_CASE": "[purple]GOLD_SLICE[/purple]",
            "FLAGGED": "[yellow]FLAGGED[/yellow]",
            "REJECTED": "[red]REJECTED[/red]",
        }.get(res.status.value, res.status.value)

        sig_label = res.social_signal.value if res.social_signal else "none"

        table.add_row(
            cid,
            status_style,
            str(res.quality_score),
            sig_label,
            f"{res.features.clipping_ratio*100:.1f}%",
            str(res.features.hesitation_pauses_count),
            f"{res.features.av_sync_drift_ms:+.1f}",
            (
                ", ".join(res.anomalies_detected)
                if res.anomalies_detected
                else "[green]None[/green]"
            ),
        )

    console.print(table)

    # Curate Benchmark
    benchmark_data = curate_benchmark(results, output_dir)
    report_file = os.path.join(output_dir, "benchmark_report.html")
    generate_html_report(benchmark_data, report_file)

    summary = benchmark_data["summary"]
    console.print("\n[bold green]✓ Benchmark Curation Complete![/bold green]")
    console.print(
        f"  • Total Clips Analyzed: [bold]{summary['total_clips_analyzed']}[/bold]"
    )
    console.print(
        f"  • Clean Baseline Clips: [bold green]{summary['clean_baseline_count']}[/bold green]"
    )
    console.print(
        f"  • Gold-Slice Edge Cases: [bold purple]{summary['gold_slice_edge_case_count']}[/bold purple]"
    )
    console.print(
        f"  • Flagged / Rejected:   [bold red]{summary['flagged_count'] + summary['rejected_count']}[/bold red]"
    )
    console.print(
        f"  • Pass Rate:            [bold]{summary['overall_pass_rate_pct']}%[/bold]"
    )
    console.print(
        f"  • SHA-256 Fingerprint:  [dim]{summary['manifest_sha256_fingerprint']}[/dim]"
    )
    if "parquet_paths" in benchmark_data:
        console.print("  • Parquet ML Splits:")
        for split, ppath in benchmark_data["parquet_paths"].items():
            console.print(f"      - {split}: [dim]{ppath}[/dim]")
    console.print(
        f"  • Interactive Report:   [bold underline cyan]{report_file}[/bold underline cyan]\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="BehaviorBench 2.0: Multimodal Quality & Benchmark Suite"
    )
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser(
        "generate", help="Generate synthetic conversation samples"
    )
    gen_parser.add_argument(
        "--output", default="./sample_data", help="Output directory for sample data"
    )

    run_parser = subparsers.add_parser(
        "benchmark", help="Run quality inspection and benchmark curation"
    )
    run_parser.add_argument(
        "--data", default="./sample_data", help="Input directory containing clips"
    )
    run_parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for manifests and reports",
    )
    run_parser.add_argument(
        "--interhuman",
        action="store_true",
        help="Use Interhuman Inter-2 strict curation preset",
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate_synthetic_dataset(args.output)
        console.print(
            f"[green]Synthetic multimodal dataset generated at: {args.output}[/green]"
        )
    elif args.command == "benchmark":
        run_pipeline(args.data, args.output, use_interhuman_preset=args.interhuman)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
