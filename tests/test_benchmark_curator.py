"""Unit tests for benchmark curation and Parquet exports."""

import os
import shutil
import tempfile
import pandas as pd
import pytest
from behaviorbench.benchmark_curator import curate_benchmark
from behaviorbench.config import (
    BehavioralFeatures,
    ClipInspectionResult,
    ClipStatus,
    SocialSignalCategory,
)


@pytest.fixture
def mock_results():
    f = BehavioralFeatures(
        duration_sec=4.0,
        speech_duration_sec=3.2,
        silence_ratio=0.20,
        hesitation_pauses_count=1,
        mean_pitch_hz=175.0,
        pitch_variance=20.0,
        clipping_ratio=0.002,
        snr_db=24.0,
        zero_crossing_rate=0.05,
        total_frames=120,
        mean_face_confidence=0.96,
        mean_luminance=110.0,
        head_pitch_variance=2.0,
        head_yaw_variance=3.0,
        blink_count=2,
        mouth_openness_variance=0.04,
        av_sync_drift_ms=10.0,
    )
    r1 = ClipInspectionResult(
        clip_id="test_01",
        status=ClipStatus.PASS_CLEAN,
        quality_score=100.0,
        anomalies_detected=[],
        features=f,
        benchmark_slice="clean_baseline",
        social_signal=SocialSignalCategory.CLEAN_BASELINE,
    )
    r2 = ClipInspectionResult(
        clip_id="test_02",
        status=ClipStatus.PASS_EDGE_CASE,
        quality_score=100.0,
        anomalies_detected=[],
        features=f,
        benchmark_slice="gold_slice_hesitation",
        social_signal=SocialSignalCategory.HESITATION_UNCERTAINTY,
    )
    return [r1, r2]


def test_curate_benchmark_exports_parquet_and_hash(mock_results):
    temp_dir = tempfile.mkdtemp()
    try:
        output = curate_benchmark(mock_results, temp_dir)

        # Check summary & hash
        assert output["summary"]["total_clips_analyzed"] == 2
        assert output["summary"]["manifest_sha256_fingerprint"] is not None
        assert len(output["summary"]["manifest_sha256_fingerprint"]) == 64

        # Check parquet files
        assert "parquet_paths" in output
        assert os.path.exists(output["parquet_paths"]["full"])
        assert os.path.exists(output["parquet_paths"]["gold_slice"])

        # Validate parquet readability
        df_read = pd.read_parquet(output["parquet_paths"]["gold_slice"])
        assert len(df_read) == 1
        assert df_read.iloc[0]["clip_id"] == "test_02"
        assert df_read.iloc[0]["social_signal"] == "hesitation_uncertainty"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
