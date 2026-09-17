"""Benchmark curation and auditability module for BehaviorBench.

Partitions inspected clips into 'Clean Baseline' and 'Gold-Slice Edge Cases',
computes dataset integrity SHA-256 fingerprint, and exports manifests.
"""

import csv
import hashlib
import json
import os
from typing import Dict, List
from behaviorbench.config import ClipInspectionResult, ClipStatus


def curate_benchmark(
    results: List[ClipInspectionResult],
    output_dir: str
) -> Dict:
    """Curate gold-slice and clean-baseline datasets, generating auditable manifest and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    clean_baseline = [r for r in results if r.benchmark_slice == "clean_baseline"]
    gold_slice = [r for r in results if r.benchmark_slice == "gold_slice_edge_case"]
    flagged_clips = [r for r in results if r.status == ClipStatus.FLAGGED]
    rejected_clips = [r for r in results if r.status == ClipStatus.REJECTED]

    total = len(results)
    pass_clean_count = len(clean_baseline)
    gold_slice_count = len(gold_slice)
    flagged_count = len(flagged_clips)
    rejected_count = len(rejected_clips)

    pass_rate = round(((pass_clean_count + gold_slice_count) / max(1, total)) * 100.0, 1)
    mean_quality = round(sum(r.quality_score for r in results) / max(1, total), 1)

    # Serialize manifest for hash generation
    manifest_records = []
    for r in results:
        manifest_records.append({
            "clip_id": r.clip_id,
            "status": r.status.value,
            "quality_score": r.quality_score,
            "benchmark_slice": r.benchmark_slice,
            "anomalies": r.anomalies_detected,
            "features": r.features.model_dump()
        })

    manifest_json_str = json.dumps(manifest_records, sort_keys=True)
    manifest_sha256 = hashlib.sha256(manifest_json_str.encode("utf-8")).hexdigest()

    summary_stats = {
        "dataset_name": "BehaviorBench-GoldSlice-v1",
        "total_clips_analyzed": total,
        "clean_baseline_count": pass_clean_count,
        "gold_slice_edge_case_count": gold_slice_count,
        "flagged_count": flagged_count,
        "rejected_count": rejected_count,
        "overall_pass_rate_pct": pass_rate,
        "mean_quality_score": mean_quality,
        "manifest_sha256_fingerprint": manifest_sha256
    }

    # Save JSON manifest
    manifest_path = os.path.join(output_dir, "benchmark_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "metadata": summary_stats,
            "records": manifest_records
        }, f, indent=2)

    # Save summary CSV
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "clip_id", "status", "benchmark_slice", "quality_score",
            "hesitation_pauses", "silence_ratio", "clipping_ratio",
            "snr_db", "head_yaw_var", "av_drift_ms", "anomalies"
        ])
        for r in results:
            writer.writerow([
                r.clip_id,
                r.status.value,
                r.benchmark_slice or "none",
                r.quality_score,
                r.features.hesitation_pauses_count,
                r.features.silence_ratio,
                r.features.clipping_ratio,
                r.features.snr_db,
                r.features.head_yaw_variance,
                r.features.av_sync_drift_ms,
                " | ".join(r.anomalies_detected) if r.anomalies_detected else "None"
            ])

    return {
        "summary": summary_stats,
        "manifest_path": manifest_path,
        "csv_path": csv_path,
        "records": manifest_records
    }
