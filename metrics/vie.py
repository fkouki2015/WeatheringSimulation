import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, argparse, re, time
from typing import List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
            # リストの場合は最初の要素をスカラーとして返す
            loaded_json["score"] = float(score_val[0])
        elif isinstance(score_val, (int, float)):
            loaded_json["score"] = float(score_val)
        else:
            raise ValueError(f"Invalid 'score' format: {score_val}")

        return loaded_json
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse JSON from output: {output_text}") from e

def parse_numeric_score(output_text: str) -> float:
    """VLM出力から単一のスコア(0-10)をパースする(数字のみ想定)"""
    if output_text is None:
        raise ValueError("Output was None")
    text = output_text.strip()
    try:
        val = float(text)
    except ValueError:
        # Fallback: extract first number from the text (in case of stray tokens/newlines).
        m = re.search(r"[-+]?(?:\d*\.\d+|\d+)", text)
        if not m:
            raise ValueError(f"Failed to parse numeric score from output: {output_text!r}")
        val = float(m.group(0))
    # Soft clamp to expected range to guard against occasional out-of-range generations
    return float(max(0.0, min(10.0, val)))


def aging_progression_score(scores: List[float]) -> Dict[str, float]:
    from scipy.stats import spearmanr

    n = len(scores)
    if n < 2:
        return {"spearman": 0.0, "monotonicity": 0.0, "smoothness": 0.0, "progression": 0.0}

    # Spearman相関 (単調増加傾向)
    rho, _ = spearmanr(range(n), scores)
    rho = float(rho) if not np.isnan(rho) else 0.0
    spearman_norm = (rho + 1.0) / 2.0  # [-1,+1] -> [0,1]

    # 単調増加率 (増加 or 維持の割合)
    mono = sum(1 for i in range(n - 1) if scores[i + 1] >= scores[i]) / (n - 1)

    # 滑らかさ: 変化量の標準偏差が小さいほど滑らか
    diffs = np.diff(scores)
    std_diffs = float(np.std(diffs))
    # 最大ありうる標準偏差は10 (スコアレンジ) で正規化
    smoothness = 1.0 / (1.0 + std_diffs / 10.0)

    progression = (spearman_norm + mono + smoothness) / 3.0

    return {
        "spearman": round(rho, 4),
        "monotonicity": round(mono, 4),
        "smoothness": round(smoothness, 4),
        "progression": round(progression, 4),
    }

# ---------------------------
# Prompts
# ---------------------------

_common_context = """
You are a professional digital artist specializing in realistic weathering and aging effects.
You will evaluate AI-generated image sequences that simulate gradual weathering/aging of objects.
All images are AI-generated, so there are no privacy concerns.

You will give your output in this way (Keep your reasoning concise and short.):
{
"score" : <integer_0_to_10>,
"reasoning" : "..."
}
"""

_prompts = {
    "weathering_strength": """
RULES:
You are given:
1. An original input image (the first image)
2. A generated weathered image (the second image)

Evaluate "Weathering Strength" (0-10):
Rate how strong the weathering effect is in the generated weathered image.

(
    0 = The image shows no weathering at all.
    10 = The weathering is sufficient.
)
""",
    "structure_preservation": """
RULES:
You are given:
1. An original input image (the first image)
2. A generated weathered image (the second image)

Evaluate "Structure Preservation" (0-10):
Rate how well the original image's structure, spatial layout, object identity, and background are preserved in the weathered image.
Do not evaluate the weathering effect itself; only evaluate if the underlying object/scene structure remains consistent.

(
    0 = The structure and background are completely changed; the object is unrecognizable or the scene has changed entirely.
    10 = The structure, object shape, spatial layout, and background are perfectly preserved; only the surface appearance changes due to weathering.
)
""",
}

_aging_prompt = """
RULES:
You are given:
1. An original input image (the first image).
2. A generated image that depicts weathering/aging.

Evaluate the "Aging Degree" (how much weathering/aging is visible) for the generated image on a scale from 0 to 10.
0 = No weathering (looks exactly like the original clean object).
10 = Maximum weathering (extremely aged, rusted, decayed, or damaged).

IMPORTANT:
- Output ONLY a single number (0 to 10).
- Do NOT output JSON.
- Do NOT output any reasoning, words, units, or extra symbols.
"""


class QwenVIEJudge:

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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

    def evaluate_weathering_strength(self, images: List[Image.Image]) -> str:
        """Weathering Naturalness を評価 (0-10, JSON形式)
        期待入力: [original_image, final_frame]
        """
        full_prompt = _common_context + _prompts["weathering_strength"]
        return self._gen(images, full_prompt)

    def evaluate_structure_preservation(self, images: List[Image.Image]) -> str:
        """Structure Preservation を評価 (0-10, JSON形式)
        期待入力: [original_image, final_frame]
        """
        full_prompt = _common_context + _prompts["structure_preservation"]
        return self._gen(images, full_prompt)

    def evaluate_aging(self, images: List[Image.Image]) -> str:
        """
        1フレームの経年変化度合いを評価 (0-10)
        期待入力: [original_image, generated_frame]
        戻り値: 数字のみのテキスト
        """
        full_prompt = _common_context + _aging_prompt
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
            "avg_weathering_strength": 0.0,
            "avg_structure_preservation": 0.0,
            "avg_spearman": 0.0,
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

            # 最終フレームのみ評価
            final_frame = frames[-1]

            # Weathering Naturalness (最終フレーム)
            nat_out = judge.evaluate_weathering_strength([original_image, final_frame])
            nat_json = parse_json_output(nat_out)
            weathering_strength = float(nat_json["score"])
            nat_reasoning = nat_json.get("reasoning", "")

            # Structure Preservation (最終フレーム)
            str_out = judge.evaluate_structure_preservation([original_image, final_frame])
            str_json = parse_json_output(str_out)
            structure_preservation = float(str_json["score"])
            str_reasoning = str_json.get("reasoning", "")

            # Aging degree evaluation (全フレーム)
            aging_scores = []
            for frame in frames:
                aging_out = judge.evaluate_aging([original_image, frame])
                aging_scores.append(parse_numeric_score(aging_out))

            # Aging progression score (段階的変化の評価)
            prog = aging_progression_score(aging_scores)

            results["samples"].append({
                "image_path": sample["image_path"],
                "output_prompt": output_prompt,
                "weathering_strength": weathering_strength,
                "structure_preservation": structure_preservation,
                "aging_spearman": prog["spearman"],
                "aging_scores": aging_scores,
                "reasoning": {
                    "weathering_strength": nat_reasoning,
                    "structure_preservation": str_reasoning,
                },
            })



        # 平均を計算
        num_samples = len(results["samples"])
        if num_samples > 0:
            results["num_samples"] = num_samples
            results["avg_weathering_strength"] = sum(s["weathering_strength"] for s in results["samples"]) / num_samples
            results["avg_structure_preservation"] = sum(s["structure_preservation"] for s in results["samples"]) / num_samples
            results["avg_spearman"] = sum(s["aging_spearman"] for s in results["samples"]) / num_samples

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
                    "avg_weathering_strength": 0.0,
                    "avg_structure_preservation": 0.0,
                    "avg_spearman": 0.0,
                    "samples": merged_samples,
                }
                
                if final_results["num_samples"] > 0:
                    n = final_results["num_samples"]
                    final_results["avg_weathering_strength"] = sum(s["weathering_strength"] for s in merged_samples) / n
                    final_results["avg_structure_preservation"] = sum(s["structure_preservation"] for s in merged_samples) / n
                    final_results["avg_spearman"] = sum(s["aging_spearman"] for s in merged_samples) / n

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
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
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
