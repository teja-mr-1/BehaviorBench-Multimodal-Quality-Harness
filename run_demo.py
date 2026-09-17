#!/usr/bin/env python3
"""Run complete BehaviorBench demonstration pipeline."""

import os
import sys

# Ensure behaviorbench package is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from behaviorbench.cli import run_pipeline
from behaviorbench.data_generator import generate_synthetic_dataset


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "sample_data")
    output_dir = os.path.join(base_dir, "output")

    print("=================================================================")
    print("      BehaviorBench: Multimodal Quality & Benchmark Suite        ")
    print("=================================================================\n")

    print(f"Step 1: Generating synthetic conversational audio/video samples in {sample_dir}...")
    manifest_path = generate_synthetic_dataset(sample_dir)
    print(f"✓ Synthetic multimodal samples generated. Manifest: {manifest_path}\n")

    print(f"Step 2: Running signal extraction, quality gatekeeper, and benchmark curation...")
    run_pipeline(sample_dir, output_dir)


if __name__ == "__main__":
    main()
