"""Interactive HTML report generator for BehaviorBench.

Generates a standalone, responsive HTML dashboard presenting benchmark metrics,
quality gate statistics, and individual clip behavioral profiles.
"""

import json
import os
from typing import Dict


def generate_html_report(benchmark_data: Dict, output_file: str) -> str:
    """Generate an interactive HTML dashboard report for BehaviorBench."""
    summary = benchmark_data["summary"]
    records = benchmark_data["records"]

    rows_html = []
    for r in records:
        status = r["status"]
        slice_name = r.get("benchmark_slice") or "—"
        score = r["quality_score"]
        feat = r["features"]
        anomalies = r.get("anomalies", [])

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

        anom_text = "<br>".join([f"⚠️ {a}" for a in anomalies]) if anomalies else "<span style='color: #10b981;'>✓ Pass (No Anomalies)</span>"

        rows_html.append(f"""
        <tr>
          <td><strong style="color: #0f172a;">{r['clip_id']}</strong></td>
          <td><span class="badge {badge_class}">{badge_text}</span></td>
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
        """)

    table_body = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>BehaviorBench - Multimodal Benchmark Report</title>
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
      max-width: 1200px;
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
      background: #fef3c7;
      color: #92400e;
      border: 1px solid #fde68a;
    }}
    .badge-rejected {{
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fecaca;
    }}
    .progress-bar-bg {{
      width: 70px;
      height: 7px;
      background: #e2e8f0;
      border-radius: 4px;
      overflow: hidden;
    }}
    .progress-bar-fill {{
      height: 100%;
      background: #2563eb;
      border-radius: 4px;
    }}
    .footer {{
      margin-top: 24px;
      text-align: center;
      font-size: 0.85rem;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title-row">
        <div>
          <h1>BehaviorBench Inspection Dashboard</h1>
          <div class="tagline">Automated Multimodal Data Quality & Regression Benchmark Suite for Non-Verbal AI</div>
        </div>
        <div class="hash-badge" title="Deterministic SHA-256 integrity hash of this benchmark slice">
          SHA-256: {summary['manifest_sha256_fingerprint'][:16]}...
        </div>
      </div>
    </div>

    <!-- KPIs -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Clips Processed</div>
        <div class="kpi-value">{summary['total_clips_analyzed']}</div>
        <div class="kpi-sub">Conversational streams</div>
      </div>
      <div class="kpi-card" style="border-left: 4px solid #10b981;">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value" style="color: #10b981;">{summary['overall_pass_rate_pct']}%</div>
        <div class="kpi-sub">{summary['clean_baseline_count']} Clean + {summary['gold_slice_edge_case_count']} Edge Cases</div>
      </div>
      <div class="kpi-card" style="border-left: 4px solid #8b5cf6;">
        <div class="kpi-label">Gold-Slice Edge Cases</div>
        <div class="kpi-value" style="color: #8b5cf6;">{summary['gold_slice_edge_case_count']}</div>
        <div class="kpi-sub">Hesitations & Head Turns</div>
      </div>
      <div class="kpi-card" style="border-left: 4px solid #ef4444;">
        <div class="kpi-label">Flagged / Rejected</div>
        <div class="kpi-value" style="color: #ef4444;">{summary['flagged_count'] + summary['rejected_count']}</div>
        <div class="kpi-sub">{summary['rejected_count']} Critical + {summary['flagged_count']} Flagged</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mean Quality Score</div>
        <div class="kpi-value">{summary['mean_quality_score']} <span style="font-size: 1rem; color: #94a3b8;">/ 100</span></div>
        <div class="kpi-sub">Across full test batch</div>
      </div>
    </div>

    <!-- Inspection Table -->
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Clip ID</th>
            <th>Validation Status</th>
            <th>Quality Score</th>
            <th>Acoustic Pauses</th>
            <th>Audio Clipping (SNR)</th>
            <th>Head Movement Variance</th>
            <th>AV Sync Drift</th>
            <th>Anomalies & Gating Verdict</th>
          </tr>
        </thead>
        <tbody>
          {table_body}
        </tbody>
      </table>
    </div>

    <div class="footer">
      BehaviorBench v0.1.0 &bull; Built by Raviteja Maddela (MSc Computer Science, DTU) &bull; Designed for Applied Multimodal ML at Interhuman AI
    </div>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_file
