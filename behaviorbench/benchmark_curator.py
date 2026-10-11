"""Benchmark curation and auditability module for BehaviorBench 2.0.

Partitions inspected clips into 'Clean Baseline', targeted 'Gold-Slice Social Signal Edge Cases',
and quarantined splits. Computes SHA-256 cryptographic dataset fingerprint,
and exports manifests in JSON, CSV, and Apache Parquet (.parquet) formats for PyTorch/Hugging Face.
"""

import csv
import hashlib
import json
import os
from typing import Dict, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from behaviorbench.config import ClipInspectionResult, ClipStatus


def curate_benchmark(
    results: List[ClipInspectionResult],
    output_dir: str,
) -> Dict:
    """Curate gold-slice and clean-baseline datasets, exporting Parquet, JSON, and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    clean_baseline = [r for r in results if r.benchmark_slice == "clean_baseline"]
    gold_slice = [
        r
        for r in results
        if r.benchmark_slice and r.benchmark_slice.startswith("gold_slice")
    ]
    flagged_clips = [r for r in results if r.status == ClipStatus.FLAGGED]
    rejected_clips = [r for r in results if r.status == ClipStatus.REJECTED]

    total = len(results)
    pass_clean_count = len(clean_baseline)
    gold_slice_count = len(gold_slice)
    flagged_count = len(flagged_clips)
    rejected_count = len(rejected_clips)

    pass_rate = round(
        ((pass_clean_count + gold_slice_count) / max(1, total)) * 100.0, 1
    )
    mean_quality = round(sum(r.quality_score for r in results) / max(1, total), 1)

    # Social signal breakdown
    signal_counts: Dict[str, int] = {}
    for r in results:
        sig_name = r.social_signal.value if r.social_signal else "unclassified"
        signal_counts[sig_name] = signal_counts.get(sig_name, 0) + 1

    # Serialize manifest records
    manifest_records = []
    flat_rows = []
    for r in results:
        f = r.features
        record = {
            "clip_id": r.clip_id,
            "status": r.status.value,
            "quality_score": r.quality_score,
            "benchmark_slice": r.benchmark_slice or "none",
            "social_signal": r.social_signal.value if r.social_signal else "none",
            "anomalies": r.anomalies_detected,
            "features": f.model_dump(),
        }
        manifest_records.append(record)

        flat_rows.append(
            {
                "clip_id": r.clip_id,
                "status": r.status.value,
                "benchmark_slice": r.benchmark_slice or "none",
                "social_signal": r.social_signal.value if r.social_signal else "none",
                "quality_score": r.quality_score,
                "duration_sec": f.duration_sec,
                "speech_duration_sec": f.speech_duration_sec,
                "silence_ratio": f.silence_ratio,
                "hesitation_pauses": f.hesitation_pauses_count,
                "mean_pitch_hz": f.mean_pitch_hz,
                "pitch_variance": f.pitch_variance,
                "clipping_ratio": f.clipping_ratio,
                "snr_db": f.snr_db,
                "zero_crossing_rate": f.zero_crossing_rate,
                "mean_face_confidence": f.mean_face_confidence,
                "mean_luminance": f.mean_luminance,
                "head_pitch_variance": f.head_pitch_variance,
                "head_yaw_variance": f.head_yaw_variance,
                "mouth_openness_variance": f.mouth_openness_variance,
                "av_sync_drift_ms": f.av_sync_drift_ms,
                "anomalies": " | ".join(r.anomalies_detected) if r.anomalies_detected else "None",
            }
        )

    manifest_json_str = json.dumps(manifest_records, sort_keys=True)
    manifest_sha256 = hashlib.sha256(manifest_json_str.encode("utf-8")).hexdigest()

    summary_stats = {
        "dataset_name": "BehaviorBench-Interhuman-GoldSlice-v2",
        "total_clips_analyzed": total,
        "clean_baseline_count": pass_clean_count,
        "gold_slice_edge_case_count": gold_slice_count,
        "flagged_count": flagged_count,
        "rejected_count": rejected_count,
        "overall_pass_rate_pct": pass_rate,
        "mean_quality_score": mean_quality,
        "social_signal_distribution": signal_counts,
        "manifest_sha256_fingerprint": manifest_sha256,
    }

    # 1. Save JSON manifest
    manifest_path = os.path.join(output_dir, "benchmark_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {"metadata": summary_stats, "records": manifest_records},
            f,
            indent=2,
        )

    # 2. Save CSV summary
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_all = pd.DataFrame(flat_rows)
    df_all.to_csv(csv_path, index=False)

    # 3. Export Apache Parquet splits for PyTorch / Hugging Face DataLoaders
    parquet_paths: Dict[str, str] = {}
    try:
        # Full benchmark dataset parquet
        full_parquet_path = os.path.join(output_dir, "benchmark_full_dataset.parquet")
        df_all.to_parquet(full_parquet_path, engine="pyarrow", index=False)
        parquet_paths["full"] = full_parquet_path

        # Gold Slice edge cases parquet split
        df_gold = df_all[df_all["benchmark_slice"].str.startswith("gold_slice")].copy()
        gold_parquet_path = os.path.join(
            output_dir, "benchmark_gold_slice_edgecases.parquet"
        )
        df_gold.to_parquet(gold_parquet_path, engine="pyarrow", index=False)
        parquet_paths["gold_slice"] = gold_parquet_path

        # Clean baseline parquet split
        df_clean = df_all[df_all["benchmark_slice"] == "clean_baseline"].copy()
        clean_parquet_path = os.path.join(
            output_dir, "benchmark_clean_baseline.parquet"
        )
        df_clean.to_parquet(clean_parquet_path, engine="pyarrow", index=False)
        parquet_paths["clean_baseline"] = clean_parquet_path

        # Quarantined / rejected parquet split
        df_quarantine = df_all[
            df_all["status"].isin([ClipStatus.FLAGGED.value, ClipStatus.REJECTED.value])
        ].copy()
        quarantine_parquet_path = os.path.join(
            output_dir, "benchmark_quarantined.parquet"
        )
        df_quarantine.to_parquet(quarantine_parquet_path, engine="pyarrow", index=False)
        parquet_paths["quarantine"] = quarantine_parquet_path

    except Exception as e:
        print(f"Warning: Parquet export encountered an error: {e}")

    return {
        "summary": summary_stats,
        "manifest_path": manifest_path,
        "csv_path": csv_path,
        "parquet_paths": parquet_paths,
        "records": manifest_records,
    }
