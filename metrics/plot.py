import json
import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn

import japanize_matplotlib

plt.rcParams.update({'font.size': 10})


def load_metrics(metrics_dir: Path, models: List[str]) -> Dict[str, List[dict]]:
    """Load metrics from JSONL files for each model."""
    all_data = {}
    for model in models:
        metrics_file = metrics_dir / f"{model}_metrics.jsonl"
        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found")
            continue
        
        with open(metrics_file, "r") as f:
            data = [json.loads(line) for line in f]
        all_data[model] = data
    
    return all_data


def plot_metrics(
    metrics_dir: str,
    models: List[str],
    output_path: str = None
):
    """
    Plot each metric vs clip_text for all models.
    
    Args:
        metrics_dir: Directory containing {model}_metrics.jsonl files
        models: List of model names to plot
        output_path: Output path for plots (default: metrics_dir)
    """
    metrics_dir = Path(metrics_dir)
    output_path = Path(output_path) if output_path else metrics_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    all_data = load_metrics(metrics_dir, models)
    
    if not all_data:
        print("No data found!")
        return
    
    seaborn.set_style("darkgrid")
    
    # Metrics to plot (y-axis)
    metrics_to_plot = [
        ("clip_image", "CLIP Image Similarity"),
        ("mse", "MSE"),
        ("lpips", "LPIPS"),
        ("ssim", "SSIM"),
        ("dino", "DINO Similarity"),
    ]
    
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "H"]
    colors = plt.cm.tab10.colors
    
    # Create individual plots for each metric
    for metric_key, metric_label in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        for i, (model_name, data) in enumerate(all_data.items()):
            if not data:
                continue
            
            xs = [d["clip_text"] for d in data]
            ys = [d[metric_key] for d in data]
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            ax.plot(xs, ys, marker=marker, linewidth=2, markersize=6, 
                   label=model_name, color=color)
        
        ax.set_xlabel("CLIP Text Similarity", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        # ax.set_title(f"{metric_label} vs CLIP Text Similarity", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        plot_path = output_path / f"{metric_key}.pdf"
        plt.savefig(plot_path, bbox_inches="tight")
        # plt.savefig(output_path / f"{metric_key}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")
    



def main():
    parser = argparse.ArgumentParser(description="Plot metrics from compute_metrics output")
    parser.add_argument("--metrics_dir", type=str, default="metrics_out",)
    parser.add_argument("--models", type=str, nargs="+", default=["proposed", "flux", "qwen", "ip2p", "sd", "sdedit"], choices=["proposed", "flux", "qwen", "ip2p", "sd", "sdedit"])
    parser.add_argument("--output_path", type=str, default="plots")
    args = parser.parse_args()
    
    plot_metrics(
        metrics_dir=args.metrics_dir,
        models=args.models,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
