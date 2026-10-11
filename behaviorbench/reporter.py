"""Interactive HTML report generator for BehaviorBench 2.0.

Generates a standalone, responsive HTML dashboard presenting benchmark metrics,
quality gate statistics, individual clip behavioral profiles, and Parquet export info.
"""

import json
import os
from typing import Dict


def generate_html_report(benchmark_data: Dict, output_file: str) -> str:
    """Generate an interactive HTML dashboard report for BehaviorBench 2.0."""
    summary = benchmark_data["summary"]
    records = benchmark_data["records"]

    rows_html = []
    for r in records:
        status = r["status"]
        score = r["quality_score"]
        feat = r["features"]
        anomalies = r.get("anomalies", [])
        sig_name = r.get("social_signal") or "none"

        if status == "PASS_CLEAN":
            badge_class = "badge-clean"
            badge_text = "Clean Baseline"
        elif status == "PASS_EDGE_CASE":
            badge_class = "badge-edge"
            badge_text = "Gold Slice (Edge Case)"
        elif status == "FLAGGED":
            badge_class = "badge-flagged"
            badge_text = "Flagged"
        else:
            badge_class = "badge-rejected"
            badge_text = "Rejected"

        # Signal badge class
        sig_style = "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;"
        if "hesitation" in sig_name:
            sig_style = "background: #fef3c7; color: #92400e; border: 1px solid #fde68a;"
        elif "agreement" in sig_name:
            sig_style = "background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe;"
        elif "expressive" in sig_name:
            sig_style = "background: #fce7f3; color: #9d174d; border: 1px solid #fbcfe8;"
        elif "reflection" in sig_name:
            sig_style = "background: #ccfbf1; color: #115e59; border: 1px solid #99f6e4;"
        elif "clean" in sig_name:
            sig_style = "background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;"

        anom_text = (
            "<br>".join([f"⚠️ {a}" for a in anomalies])
            if anomalies
            else "<span style='color: #10b981;'>✓ Pass (No Anomalies)</span>"
        )

        rows_html.append(
            f"""
        <tr>
          <td><strong style="color: #0f172a;">{r['clip_id']}</strong></td>
          <td><span class="badge {badge_class}">{badge_text}</span></td>
          <td><span class="badge" style="{sig_style}">{sig_name}</span></td>
          <td>
            <div style="display: flex; align-items: center; gap: 8px;">
              <strong>{score}</strong>
              <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: {score}%;"></div></div>
            </div>
          </td>
          <td>{feat['hesitation_pauses_count']} pauses ({feat['silence_ratio']*100:.0f}% silence)</td>
          <td>{feat['clipping_ratio']*100:.2f}% (SNR: {feat['snr_db']} dB)</td>
          <td>Yaw: {feat['head_yaw_variance']:.1f}° | Pitch: {feat['head_pitch_variance']:.1f}°</td>
          <td>{feat['av_sync_drift_ms']:+.1f} ms</td>
          <td style="font-size: 0.82rem;">{anom_text}</td>
        </tr>
        """
        )

    table_body = "\n".join(rows_html)

    # Parquet list html
    parquet_items = ""
    if "parquet_paths" in benchmark_data:
        for split_name, path in benchmark_data["parquet_paths"].items():
            base_name = os.path.basename(path)
            parquet_items += f"""<span class="parquet-badge">📦 {split_name}: {base_name}</span> """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>BehaviorBench 2.0 - Multimodal Benchmark Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      padding: 28px 20px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
    }}
    .header {{
      background: #ffffff;
      padding: 24px 30px;
      border-radius: 12px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      margin-bottom: 24px;
    }}
    .title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    h1 {{
      font-size: 1.7rem;
      font-weight: 800;
      color: #0f172a;
      letter-spacing: -0.5px;
    }}
    .tagline {{
      font-size: 0.92rem;
      color: #64748b;
      margin-top: 4px;
    }}
    .hash-badge {{
      font-family: monospace;
      font-size: 0.78rem;
      background: #f1f5f9;
      color: #475569;
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid #cbd5e1;
    }}
    .parquet-row {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .parquet-badge {{
      font-family: monospace;
      font-size: 0.75rem;
      background: #eff6ff;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
      padding: 4px 10px;
      border-radius: 6px;
    }}
    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: #ffffff;
      padding: 20px;
      border-radius: 10px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .kpi-label {{
      font-size: 0.8rem;
      text-transform: uppercase;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.5px;
    }}
    .kpi-value {{
      font-size: 1.8rem;
      font-weight: 800;
      color: #0f172a;
      margin-top: 6px;
    }}
    .kpi-sub {{
      font-size: 0.8rem;
      color: #94a3b8;
      margin-top: 2px;
    }}
    /* Table */
    .table-card {{
      background: #ffffff;
      border-radius: 12px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.88rem;
    }}
    th {{
      background: #f8fafc;
      color: #475569;
      font-weight: 700;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding: 14px 16px;
      border-bottom: 1px solid #e2e8f0;
    }}
    td {{
      padding: 13px 16px;
      border-bottom: 1px solid #f1f5f9;
      color: #334155;
    }}
    tr:hover td {{
      background: #f8fafc;
    }}
    /* Badges */
    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 5px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    .badge-clean {{
      background: #dcfce7;
      color: #166534;
      border: 1px solid #bbf7d0;
    }}
    .badge-edge {{
      background: #f3e8ff;
      color: #6b21a8;
      border: 1px solid #e9d5ff;
    }}
    .badge-flagged {{
      background: #fef9c3;
      color: #854d0e;
      border: 1px solid #fef08a;
    }}
    .badge-rejected {{
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fecaca;
    }}
    /* Progress Bar */
    .progress-bar-bg {{
      width: 60px;
      height: 7px;
      background: #e2e8f0;
      border-radius: 4px;
      overflow: hidden;
    }}
    .progress-bar-fill {{
      height: 100%;
      background: #10b981;
      border-radius: 4px;
    }}
    .footer {{
      margin-top: 24px;
      text-align: center;
      font-size: 0.82rem;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title-row">
        <div>
          <h1>BehaviorBench 2.0 🎯</h1>
          <div class="tagline">Multimodal Quality Gatekeeper & "Gold Slice" Social Signal Benchmark Platform</div>
        </div>
        <div>
          <span class="hash-badge">SHA-256: {summary['manifest_sha256_fingerprint'][:16]}...</span>
        </div>
      </div>
      <div class="parquet-row">
        <span style="font-size: 0.82rem; font-weight: 600; color: #475569;">ML Training Splits:</span>
        {parquet_items}
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Clips</div>
        <div class="kpi-value">{summary['total_clips_analyzed']}</div>
        <div class="kpi-sub">Conversational streams</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Clean Baseline</div>
        <div class="kpi-value" style="color: #166534;">{summary['clean_baseline_count']}</div>
        <div class="kpi-sub">Passed all gates</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Gold-Slice Edge Cases</div>
        <div class="kpi-value" style="color: #6b21a8;">{summary['gold_slice_edge_case_count']}</div>
        <div class="kpi-sub">Targeted social signals</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Quarantined / Rejected</div>
        <div class="kpi-value" style="color: #dc2626;">{summary['flagged_count'] + summary['rejected_count']}</div>
        <div class="kpi-sub">Blocked from training</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value" style="color: #0284c7;">{summary['overall_pass_rate_pct']}%</div>
        <div class="kpi-sub">Mean Score: {summary['mean_quality_score']} / 100</div>
      </div>
    </div>

    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Clip ID</th>
            <th>Gate Status</th>
            <th>Social Signal</th>
            <th>Quality Score</th>
            <th>Speech / Pauses</th>
            <th>Audio Acoustics</th>
            <th>Head Kinetics</th>
            <th>AV Sync Drift</th>
            <th>Anomalies / Gate Result</th>
          </tr>
        </thead>
        <tbody>
          {table_body}
        </tbody>
      </table>
    </div>

    <div class="footer">
      Generated automatically by <strong>BehaviorBench 2.0</strong> &bull; DTU MSc Computer Science (AI & Data Science)
    </div>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(html_content)

    return output_file
