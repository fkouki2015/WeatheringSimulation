#!/usr/bin/env python3
"""
vie_*.json を読み込んで全モデルの指標を横並びの表にまとめるスクリプト。

Usage:
    python summarize_vie.py <vie_json_dir>

Example:
    python summarize_vie.py /work/DDIPM/kfukushima/wsim/images_out_old2/output_def
"""

import json
import sys
from pathlib import Path


# 表示したい指標キーと表示名のマッピング（存在しないキーは自動スキップ）
# spearman は [-1, +1] -> [0, 10] に変換して表示
METRIC_KEYS = [
    ("avg_faithfulness",          "Faithfulness"),
    ("avg_consistency",           "Consistency"),
    ("avg_semantic",              "Semantic"),
    ("avg_naturalness",           "Naturalness"),
    ("avg_artifacts",             "Artifacts"),
    ("avg_aesthetic",             "Aesthetic"),
    ("avg_weathering_strength",   "Wthr.Strength"),
    ("avg_structure_preservation","Struct.Prsrv"),
    # ("avg_spearman",              "Spearman"),
    # ("avg_monotonicity",          "Monotonicity"),
    # ("avg_smoothness",            "Smoothness"),
    # ("avg_progression",           "Progression"),
]

# Spearman は [-1, +1] -> [0, 10] に変換
SPEARMAN_KEY = "avg_spearman"


def spearman_to_10(v: float) -> float:
    """Spearman相関 [-1, +1] を [0, 10] スケールに変換"""
    return (v + 1.0) / 2.0 * 10.0


def load_results(json_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(json_dir.glob("vie_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        row = {"model": data.get("model", path.stem), "num_samples": data.get("num_samples", "?")}
        for key, _ in METRIC_KEYS:
            v = data.get(key, None)
            if v is not None and key == SPEARMAN_KEY:
                v = spearman_to_10(v)
            row[key] = v
        rows.append(row)
    return rows


def print_table(rows: list[dict], csv_dir: Path):
    if not rows:
        print("No vie_*.json files found.")
        return

    # 実際に存在する指標だけ使う
    active_metrics = [(k, label) for k, label in METRIC_KEYS
                      if any(r.get(k) is not None for r in rows)]

    # 全モデル平均行を追加
    avg_row = {"model": "AVERAGE", "num_samples": sum(r["num_samples"] for r in rows if isinstance(r["num_samples"], int))}
    for key, _ in active_metrics:
        vals = [r[key] for r in rows if r.get(key) is not None]
        avg_row[key] = sum(vals) / len(vals) if vals else None

    display_rows = rows + [None, avg_row]  # None は区切り線

    # カラム幅計算
    model_w = max(len(r["model"]) for r in rows + [avg_row])
    model_w = max(model_w, 7)
    col_w = 14

    header = f"{'Model':<{model_w}}  {'N':>4}" + "".join(f"  {label:>{col_w}}" for _, label in active_metrics) + f"  {'AVG':>{col_w}}"
    sep = "-" * len(header)

    def row_avg(r):
        vals = [r[k] for k, _ in active_metrics if r.get(k) is not None]
        return sum(vals) / len(vals) if vals else None

    print(sep)
    print(header)
    print(sep)

    for r in display_rows:
        if r is None:
            print(sep)
            continue
        line = f"{r['model']:<{model_w}}  {r['num_samples']:>4}"
        for key, _ in active_metrics:
            val = r.get(key)
            if val is None:
                line += f"  {'N/A':>{col_w}}"
            else:
                line += f"  {val:>{col_w}.4f}"
        avg = row_avg(r)
        line += f"  {avg:>{col_w}.4f}" if avg is not None else f"  {'N/A':>{col_w}}"
        print(line)

    print(sep)

    # CSV出力
    csv_path = csv_dir / "vie_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header_csv = "model,num_samples," + ",".join(k for k, _ in active_metrics)
        f.write(header_csv + "\n")
        for r in rows + [avg_row]:
            vals = [r["model"], str(r["num_samples"])]
            for key, _ in active_metrics:
                v = r.get(key)
                vals.append(f"{v:.4f}" if v is not None else "")
            f.write(",".join(vals) + "\n")
    print(f"\nCSV saved: {csv_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <vie_json_dir>")
        sys.exit(1)

    json_dir = Path(sys.argv[1])
    if not json_dir.is_dir():
        print(f"Error: {json_dir} is not a directory.")
        sys.exit(1)

    rows = load_results(json_dir)
    print_table(rows, json_dir)


if __name__ == "__main__":
    main()
