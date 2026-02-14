import sys
from argparse import ArgumentParser
from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image

import json
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path

import lpips
from pytorch_msssim import ssim
from clip_similarity import ClipSimilarity


def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute MSE between two images. Images should be in [-1, 1] range."""
    return F.mse_loss(img1, img2).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two images. Images should be in [0, 1] range."""
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    return ssim(img1, img2, data_range=1.0, size_average=True).item()


class DinoSimilarity(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = (images + 1) / 2
        images = (images - self.mean) / self.std
        images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
        return self.model(images)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        feat1 = self.encode(img1)
        feat2 = self.encode(img2)
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        return (feat1 * feat2).sum(dim=-1)


def load_gif_frames(gif_path: str) -> List[Image.Image]:
    """Load all frames from a GIF file."""
    frames = []
    with Image.open(gif_path) as gif:
        for frame_idx in range(gif.n_frames):
            gif.seek(frame_idx)
            frames.append(gif.convert("RGB").copy())
    return frames


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor in [-1, 1] range."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    return tensor * 2 - 1  # [0, 1] -> [-1, 1]


class MetricsEvaluator:
    """Evaluator for computing image quality metrics."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading CLIP similarity model...")
        self.clip_similarity = ClipSimilarity().to(device)
        print("Loading LPIPS model...")
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        print("Loading DINO model...")
        self.dino_similarity = DinoSimilarity().to(device)
    
    def evaluate_frame(
        self, 
        input_tensor: torch.Tensor, 
        gen_tensor: torch.Tensor,
        input_prompt: str,
        output_prompt: str
    ) -> dict:
        """Evaluate a single generated frame against the input image."""
        input_tensor = input_tensor.to(self.device)
        gen_tensor = gen_tensor.to(self.device)
        
        # CLIP similarities
        _, clip_text, clip_direction, clip_image = self.clip_similarity(
            input_tensor, gen_tensor, [input_prompt], [output_prompt]
        )
        
        # MSE
        mse_val = compute_mse(input_tensor, gen_tensor)
        
        # LPIPS
        lpips_val = self.lpips_model(input_tensor, gen_tensor).item()
        
        # SSIM
        ssim_val = compute_ssim(input_tensor, gen_tensor)
        
        # DINO
        dino_val = self.dino_similarity(input_tensor, gen_tensor).item()
        
        return {
            "clip_text": clip_text.item(),
            "clip_direction": clip_direction.item(),
            "clip_image": clip_image.item(),
            "mse": mse_val,
            "lpips": lpips_val,
            "ssim": ssim_val,
            "dino": dino_val,
        }


def compute_metrics(
    json_path: str,
    gif_dir: str,
    output_path: str,
    models: List[str],
    device: str = "cuda"
):
    """
    Compute metrics for generated GIF files.
    
    Args:
        json_path: Path to JSON file with sample information
        gif_dir: Directory containing model subdirectories with GIF files
        output_path: Directory to save metric results
        models: List of model names to evaluate (e.g., ["flux", "ip2p", "sd", "sdedit"])
        device: Device to use for computation
    """
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = Path(gif_dir)
    
    # Initialize evaluator
    evaluator = MetricsEvaluator(device)
    
    # Process each model
    for model_name in models:
        model_gif_dir = gif_dir / model_name
        if not model_gif_dir.exists():
            print(f"Skipping {model_name}: directory not found at {model_gif_dir}")
            continue
        
        print(f"\nEvaluating {model_name}...")
        model_output_file = output_dir / f"{model_name}_metrics.jsonl"
        
        all_frame_metrics = []  # List of lists: [sample][frame] -> metrics
        
        for sample_idx, sample in enumerate(tqdm(data, desc=f"Processing {model_name}")):
            # Use input image filename with .gif extension
            input_filename = Path(sample["image_path"]).stem
            gif_path = model_gif_dir / f"{input_filename}.gif"
            
            if not gif_path.exists():
                print(f"  Warning: GIF not found: {gif_path}")
                continue
            
            # Load input image and resize to 512x512
            input_image = Image.open(sample["image_path"]).convert("RGB")
            input_image = input_image.resize((512, 512), Image.LANCZOS)
            input_tensor = pil_to_tensor(input_image)[None]  # Add batch dim
            
            # Load generated frames from GIF
            gen_frames = load_gif_frames(str(gif_path))
            
            # Evaluate each frame
            frame_metrics = []
            for frame_idx, gen_frame in enumerate(gen_frames):
                # Resize generated frame to 512x512
                gen_frame = gen_frame.resize((512, 512), Image.LANCZOS)
                gen_tensor = pil_to_tensor(gen_frame)[None]
                
                metrics = evaluator.evaluate_frame(
                    input_tensor,
                    gen_tensor,
                    sample["input_prompt"],
                    sample["output_prompt"]
                )
                metrics["sample_idx"] = sample_idx
                metrics["frame_idx"] = frame_idx
                frame_metrics.append(metrics)
            
            all_frame_metrics.append(frame_metrics)
        
        # Compute per-frame averages across all samples
        if not all_frame_metrics:
            continue
            
        num_frames = len(all_frame_metrics[0])
        
        with open(model_output_file, "w") as f:
            for frame_idx in range(num_frames):
                frame_data = [sample_metrics[frame_idx] for sample_metrics in all_frame_metrics 
                             if frame_idx < len(sample_metrics)]
                
                avg_metrics = {
                    "frame_idx": frame_idx,
                    "model": model_name,
                    "num_samples": len(frame_data),
                    "clip_text": np.mean([d["clip_text"] for d in frame_data]),
                    "clip_direction": np.mean([d["clip_direction"] for d in frame_data]),
                    "clip_image": np.mean([d["clip_image"] for d in frame_data]),
                    "mse": np.mean([d["mse"] for d in frame_data]),
                    "lpips": np.mean([d["lpips"] for d in frame_data]),
                    "ssim": np.mean([d["ssim"] for d in frame_data]),
                    "dino": np.mean([d["dino"] for d in frame_data]),
                }
                f.write(json.dumps(avg_metrics) + "\n")
        
        print(f"  Saved metrics to {model_output_file}")
    
    # Plot comparison
    plot_model_comparison(output_dir, models)


def plot_model_comparison(output_dir: Path, models: List[str] = None):
    """Generate comparison plots for all models found in output_dir."""
    plt.rcParams.update({'font.size': 11.5})
    seaborn.set_style("darkgrid")
    
    # Discover all models from metrics files in output_dir
    all_metrics_files = sorted(output_dir.glob("*_metrics.jsonl"))
    discovered_models = [f.stem.replace("_metrics", "") for f in all_metrics_files]
    if not discovered_models:
        print("No metrics files found for comparison plot.")
        return
    
    print(f"Plotting comparison for models: {discovered_models}")
    
    metrics_names = [
        ("clip_text", "CLIP Text Similarity"),
        ("clip_direction", "CLIP Direction"),
        ("clip_image", "CLIP Image Similarity"),
        ("mse", "MSE"),
        ("lpips", "LPIPS"),
        ("ssim", "SSIM"),
        ("dino", "DINO Similarity"),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=150)
    
    for model_name in discovered_models:
        metrics_file = output_dir / f"{model_name}_metrics.jsonl"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file, "r") as f:
            data = [json.loads(line) for line in f]
        
        frames = [d["frame_idx"] for d in data]
        
        for idx, (key, label) in enumerate(metrics_names):
            ax = axes[idx // 3, idx % 3]
            values = [d[key] for d in data]
            ax.plot(frames, values, marker='o', linewidth=2, markersize=4, label=model_name)
            ax.set_xlabel("Frame", labelpad=10)
            ax.set_ylabel(label, labelpad=10)
            ax.set_title(label)
    
    # Add legend to first plot
    axes[0, 0].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    print(f"Saved comparison plots to {output_dir}")


def main():
    parser = ArgumentParser(description="Evaluate generated images from competitive models")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--gif_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./metrics_out")
    parser.add_argument("--models", type=str, nargs="+", required=True, choices=["proposed", "sdedit", "sd", "flux", "qwen", "ip2p", "flowedit"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    compute_metrics(
        json_path=args.json_path,
        gif_dir=args.gif_dir,
        output_path=args.output_path,
        models=args.models,
        device=args.device
    )


if __name__ == "__main__":
    main()
