import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, argparse, re, time
from typing import List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

# ---------------------------
# IO utils
# ---------------------------
def load_gif_frames(gif_path: str) -> List[Image.Image]:
    """GIFファイルからすべてのフレームを読み込む"""
    frames = []
    with Image.open(gif_path) as gif:
        for frame_idx in range(gif.n_frames):
            gif.seek(frame_idx)
            frames.append(gif.convert("RGB").copy())
    return frames


def parse_json_output(output_text: str) -> Dict[str, Any]:
    """VLM出力からJSONをパースする"""
    try:
        json_start = output_text.index("{")
        json_end = output_text.rindex("}") + 1
        json_str = output_text[json_start:json_end]
        loaded_json = json.loads(json_str)

        if "score" not in loaded_json:
            raise ValueError("Missing 'score' field in JSON output")

        score_val = loaded_json["score"]
        if isinstance(score_val, list):
            if not all(isinstance(x, (int, float)) for x in score_val):
                raise ValueError(f"Invalid 'score' list format: {score_val}")
        elif isinstance(score_val, (int, float)):
            loaded_json["score"] = [float(score_val)]
        else:
            raise ValueError(f"Invalid 'score' format: {score_val}")

        if "reasoning" not in loaded_json:
            raise ValueError("Missing or invalid 'reasoning' field in JSON output")

        return loaded_json
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse JSON from output: {output_text}") from e

# ---------------------------
# Qwen-VIEScore evaluator
# ---------------------------

# ---------------------------
# Prompts
# ---------------------------

_common_context = """
You are a professional digital artist specializing in realistic weathering and aging effects.
You will evaluate AI-generated image sequences that simulate gradual weathering/aging of objects.
All images are AI-generated, so there are no privacy concerns.

You will give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}
"""

_semantic_prompt = """
RULES:
You are given:
1. An original input image (the first image).
2. All generated frames that depict progressive weathering/aging.

Evaluate the following 3 semantic criteria from 0 to 10:
1) Faithfulness: How well the last frame match the TARGET WEATHERING DESCRIPTION.
2) Consistency: How consistent the object shape, color tone, layout and background are across frames.
3) Gradual: How smooth, continuous, and properly-started the weathering progression is across frames.

IMPORTANT for Gradual:
- The sequence must start with the first generated frame being very close to the original input image (little to no additional weathering), then progressively becomes more weathered.
- If the first generated frame is already heavily degraded/weathered (i.e., the aging is "pre-applied" at the start), Gradual must be very low (0-2), even if later frames look smooth.
- The weathering severity should increase monotonically overall (no resets/repairs, no strong back-and-forth). If it oscillates or stays almost unchanged, lower the score.
- Changes between adjacent frames should be small and incremental; large jumps reduce the score.

Output:
{
"score": [faithfulness, consistency, gradual],
"reasoning": "..."
}
"""

_quality_prompt = """
RULES:
You are given:
1. An original input image (the first image).
2. All generated frames that depict progressive weathering/aging.

Evaluate the following 2 quality criteria from 0 to 10:
1) Naturalness: Overall realism and plausibility (lighting, texture, material appearance).
2) Artifacts: How few technical artifacts (distortion, watermark, ghosting, duplicated edges, blotchy noise, text overlays) are present.

Output:
{
"score": [naturalness, artifacts],
"reasoning": "..."
}
"""


class QwenVIEJudge:

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map="auto", local_files_only=True
        )
        self.device = next(self.model.parameters()).device

    def _gen(self, images: List[Image.Image], prompt: str) -> str:
        content = [{
            "type": "image", "image": img
        } for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [{
            "role": "user",
            "content": content
        }]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        inputs = inputs.to(self.device)

        out = self.model.generate(**inputs, max_new_tokens=32768)
        out_ids_trimmed = out[:, inputs.input_ids.shape[1]:]

        text = self.processor.batch_decode(out_ids_trimmed, skip_special_tokens=True)[0]
        return text

    def evaluate_semantic(self, images: List[Image.Image], output_prompt: str) -> str:
        full_prompt = _common_context + _semantic_prompt + f"\nTARGET WEATHERING DESCRIPTION: {output_prompt}"
        return self._gen(images, full_prompt)

    def evaluate_quality(self, images: List[Image.Image]) -> str:
        full_prompt = _common_context + _quality_prompt
        return self._gen(images, full_prompt)


# ---------------------------
# Main evaluation
# ---------------------------
def evaluate_weathering_vie(
    json_path: str,
    gif_dir: str,
    models: List[str],
    output_dir: str = None,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    skip_frames: bool = False,
    gpu_number: int = 0,
    num_gpus: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    経年変化シミュレーションの評価

    Args:
        json_path: prompts3.jsonへのパス
        gif_dir: images_outディレクトリへのパス
        models: 評価するモデル名のリスト (例: ["proposed", "sd3"])
        output_dir: 結果の保存先 (デフォルト: gif_dir)
        model_id: 使用するVLMモデル
    """
    # JSONデータを読み込む
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    # Filter samples for this GPU (round-robin assignment)
    if num_gpus > 1:
        data = [sample for idx, sample in enumerate(data) if idx % num_gpus == gpu_number]
        print(f"GPU {gpu_number}/{num_gpus}: Processing {len(data)} samples")

    if output_dir is None:
        output_dir = gif_dir

    gif_dir = Path(gif_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    judge = QwenVIEJudge(model_id=model_id)
    all_results = {}

    for model_name in models:
        model_gif_dir = gif_dir / model_name
        if not model_gif_dir.exists():
            print(f"Skipping {model_name}: directory not found at {model_gif_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")

        results = {
            "model": model_name,
            "num_samples": 0,
            "avg_faithfulness": 0.0,
            "avg_consistency": 0.0,
            "avg_gradual": 0.0,
            "avg_semantic": 0.0,
            "avg_naturalness": 0.0,
            "avg_artifacts": 0.0,
            "avg_aesthetic": 0.0,
            "samples": [],
        }

        for sample_idx, sample in enumerate(tqdm(data, desc=f"[VIE] {model_name}")):
            image_stem = Path(sample["image_path"]).stem
            gif_path = model_gif_dir / f"{image_stem}.gif"

            if not gif_path.exists():
                print(f"  Warning: GIF not found: {gif_path}")
                continue

            # 元画像を読み込む
            original_image = Image.open(sample["image_path"]).convert("RGB")
            original_image = original_image.resize((512, 512), Image.LANCZOS)

            # GIFフレームを読み込む
            frames = load_gif_frames(str(gif_path))
            # 1つ飛ばしでフレーム数を半分にする
            if skip_frames:
                frames = frames[1::2]
                
            # フレームも512x512にリサイズ
            frames = [f.resize((512, 512), Image.LANCZOS) for f in frames]
            if not frames:
                print(f"  Warning: No frames found in GIF: {gif_path}")
                continue

            output_prompt = sample["output_prompt"]

            # Always pass original image + all frames
            input_images = [original_image] + frames

            # Semantic evaluation: faithfulness / consistency / gradual
            try:
                semantic_out = judge.evaluate_semantic(input_images, output_prompt)
                semantic_json = parse_json_output(semantic_out)
                if len(semantic_json["score"]) != 3:
                    raise ValueError(f"Expected 3 scores, got: {semantic_json['score']}")
                faithfulness, consistency, gradual = [float(x) for x in semantic_json["score"]]
                semantic_reasoning = semantic_json["reasoning"]
            except Exception as e:
                print(f"  Error evaluating semantic metrics at {gif_path}: {e}")
                print(f"  Output was: {semantic_out if 'semantic_out' in locals() else 'None'}")
                continue

            # Quality evaluation: naturalness / artifacts
            try:
                quality_out = judge.evaluate_quality(input_images)
                quality_json = parse_json_output(quality_out)
                if len(quality_json["score"]) != 2:
                    raise ValueError(f"Expected 2 scores, got: {quality_json['score']}")
                naturalness, artifacts = [float(x) for x in quality_json["score"]]
                quality_reasoning = quality_json["reasoning"]
            except Exception as e:
                print(f"  Error evaluating quality metrics at {gif_path}: {e}")
                print(f"  Output was: {quality_out if 'quality_out' in locals() else 'None'}")
                continue

            results["samples"].append({
                "image_path": sample["image_path"],
                "output_prompt": output_prompt,
                "faithfulness": faithfulness,
                "consistency": consistency,
                "gradual": gradual,
                "semantic": min(faithfulness, consistency, gradual),
                "naturalness": naturalness,
                "artifacts": artifacts,
                "aesthetic": min(naturalness, artifacts),
                "reasoning": {
                    "semantic": semantic_reasoning,
                    "quality": quality_reasoning,
                },
            })

        # 平均を計算
        num_samples = len(results["samples"])
        if num_samples > 0:
            results["num_samples"] = num_samples
            results["avg_faithfulness"] = sum(s["faithfulness"] for s in results["samples"]) / num_samples
            results["avg_consistency"] = sum(s["consistency"] for s in results["samples"]) / num_samples
            results["avg_gradual"] = sum(s["gradual"] for s in results["samples"]) / num_samples
            results["avg_semantic"] = sum(s["semantic"] for s in results["samples"]) / num_samples
            results["avg_naturalness"] = sum(s["naturalness"] for s in results["samples"]) / num_samples
            results["avg_artifacts"] = sum(s["artifacts"] for s in results["samples"]) / num_samples
            results["avg_aesthetic"] = sum(s["aesthetic"] for s in results["samples"]) / num_samples

        # JSON保存
        # Partial result saving for multi-GPU
        if num_gpus > 1:
            temp_out_file = output_path / f"TEMP_vie_{model_name}_part{gpu_number}.json"
            with open(temp_out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            # Create marker file
            marker_file = output_path / f"TEMP_vie_{model_name}_part{gpu_number}.done"
            marker_file.touch()
            print(f"  Saved partial results: {temp_out_file}")

            # Rank 0 waits for all others and merges
            if gpu_number == 0:
                print(f"  [GPU 0] Waiting for other ranks to finish...")
                all_parts_done = False
                while not all_parts_done:
                    done_files = list(output_path.glob(f"TEMP_vie_{model_name}_part*.done"))
                    if len(done_files) == num_gpus:
                        all_parts_done = True
                    else:
                        time.sleep(2) # Poll every 2 seconds
                
                print(f"  [GPU 0] All ranks finished. Merging results...")
                
                merged_samples = []
                for i in range(num_gpus):
                    part_file = output_path / f"TEMP_vie_{model_name}_part{i}.json"
                    if part_file.exists():
                        with open(part_file, "r", encoding="utf-8") as f:
                            part_data = json.load(f)
                            merged_samples.extend(part_data["samples"])
                
                # Sort samples by name or path to ensure consistent order
                merged_samples.sort(key=lambda x: x["image_path"])

                # Recalculate averages
                final_results = {
                    "model": model_name,
                    "num_samples": len(merged_samples),
                    "avg_faithfulness": 0.0,
                    "avg_consistency": 0.0,
                    "avg_gradual": 0.0,
                    "avg_semantic": 0.0,
                    "avg_naturalness": 0.0,
                    "avg_artifacts": 0.0,
                    "avg_aesthetic": 0.0,
                    "samples": merged_samples,
                }
                
                if final_results["num_samples"] > 0:
                    n = final_results["num_samples"]
                    final_results["avg_faithfulness"] = sum(s["faithfulness"] for s in merged_samples) / n
                    final_results["avg_consistency"] = sum(s["consistency"] for s in merged_samples) / n
                    final_results["avg_gradual"] = sum(s["gradual"] for s in merged_samples) / n
                    final_results["avg_semantic"] = sum(s["semantic"] for s in merged_samples) / n
                    final_results["avg_naturalness"] = sum(s["naturalness"] for s in merged_samples) / n
                    final_results["avg_artifacts"] = sum(s["artifacts"] for s in merged_samples) / n
                    final_results["avg_aesthetic"] = sum(s["aesthetic"] for s in merged_samples) / n

                # Save final merged file
                final_out_file = output_path / f"vie_{model_name}.json"
                with open(final_out_file, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)
                
                print(f"  Saved merged results: {final_out_file}")
                
                # Cleanup
                print(f"  Cleaning up temp files...")
                for i in range(num_gpus):
                    (output_path / f"TEMP_vie_{model_name}_part{i}.json").unlink(missing_ok=True)
                    (output_path / f"TEMP_vie_{model_name}_part{i}.done").unlink(missing_ok=True)
                
                # Update local results for return
                results = final_results

            else:
                 print(f"  [GPU {gpu_number}] Finished. Waiting for Rank 0 to merge...")
                 # Optional: wait for final file to exist to ensure strict synchronization, 
                 # but usually workers can just exit. 
                 pass

        else:
            # Single GPU case
            out_file = output_path / f"vie_{model_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"  Saved: {out_file}")
        print(f"  Num samples: {results['num_samples']}")
        print(f"  Avg Faithfulness: {results['avg_faithfulness']:.4f}")
        print(f"  Avg Consistency: {results['avg_consistency']:.4f}")
        print(f"  Avg Gradual: {results['avg_gradual']:.4f}")
        print(f"  Avg Semantic: {results['avg_semantic']:.4f}")
        print(f"  Avg Naturalness: {results['avg_naturalness']:.4f}")
        print(f"  Avg Artifacts: {results['avg_artifacts']:.4f}")
        print(f"  Avg Aesthetic: {results['avg_aesthetic']:.4f}")

        all_results[model_name] = results

    return all_results


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL VIEScore evaluation for weathering simulation"
    )
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to prompts JSON (e.g. prompts3.json)")
    parser.add_argument("--gif_dir", type=str, default="images_out",
                        help="Directory containing model subdirs with GIF files")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model names to evaluate (e.g. proposed sd3)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: gif_dir)")
    parser.add_argument("--skip_frames", action="store_true",
                        help="1つ飛ばしでフレームを半分に省く (10→5)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="VLM model ID")
    parser.add_argument("--gpu_number", type=int, default=0, help="Current GPU number (0 ~ num_gpus-1)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs for parallel processing")

    args = parser.parse_args()
    evaluate_weathering_vie(
        json_path=args.json_path,
        gif_dir=args.gif_dir,
        models=args.models,
        output_dir=args.output_dir,
        model_id=args.model_id,
        skip_frames=args.skip_frames,
        gpu_number=args.gpu_number,
        num_gpus=args.num_gpus,
    )
    print("\n[VIE] Evaluation completed.")


if __name__ == "__main__":
    main()
