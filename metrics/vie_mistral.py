import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, argparse, re, time
import base64
from io import BytesIO
from typing import List, Tuple, Dict, Any
from PIL import Image
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

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
    cleaned = output_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    def _normalize_score(val: Any) -> float:
        if isinstance(val, list):
            if not val:
                raise ValueError("Empty score list")
            val = val[0]
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            m = re.search(r"-?\d+(?:\.\d+)?", val)
            if m:
                return float(m.group(0))
        raise ValueError(f"Invalid 'score' format: {val}")

    # First, try strict JSON parsing from the object span.
    try:
        json_start = cleaned.index("{")
        json_end = cleaned.rindex("}") + 1
        json_str = cleaned[json_start:json_end]
        loaded_json = json.loads(json_str)

        if "score" not in loaded_json:
            raise ValueError("Missing 'score' field in JSON output")
        if "reasoning" not in loaded_json:
            raise ValueError("Missing 'reasoning' field in JSON output")

        loaded_json["score"] = _normalize_score(loaded_json["score"])
        loaded_json["reasoning"] = str(loaded_json["reasoning"]).strip()
        return loaded_json
    except Exception:
        pass

    # Fallback: tolerant extraction for non-strict JSON from VLM output.
    score_match = re.search(
        r'"score"\s*:\s*(\[[^\]]+\]|-?\d+(?:\.\d+)?|"(?:[^"\\]|\\.)*")',
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not score_match:
        raise ValueError(f"Failed to parse JSON from output: {output_text}")

    raw_score = score_match.group(1).strip()
    try:
        score_val = json.loads(raw_score)
    except Exception:
        score_val = raw_score.strip('"')
    score = _normalize_score(score_val)

    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"([\s\S]*?)"\s*(?:,?\s*}|$)',
        cleaned,
        flags=re.IGNORECASE,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1)
        reasoning = reasoning.replace('\\"', '"').replace("\\n", "\n").strip()
    else:
        # Last fallback: take text after "reasoning:"
        tail_match = re.search(r'"reasoning"\s*:\s*([\s\S]+)', cleaned, flags=re.IGNORECASE)
        reasoning = tail_match.group(1).strip() if tail_match else ""
        reasoning = reasoning.rstrip("}").strip().strip('"').strip()

    if reasoning == "":
        raise ValueError(f"Failed to parse JSON from output: {output_text}")

    return {"score": score, "reasoning": reasoning}

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
    "weathering_naturalness": """
RULES:
You are given:
1. An original input image (the first image)
2. A generated weathered image (the second image)

Evaluate "Weathering Naturalness" (0-10):
Rate how natural the generated weathered image is.
Focus on the final appearance and the type of weathering.

(
    0 = The image shows no weathering at all.
    5 = Some weathering is visible but the weathering is not natural or insufficient.
    10 = The weathering is natural and sufficient.
)
""",
    "gradual_weathering": """
RULES:
You are given:
1. An original input image (the first image)
2. A sequence of generated frames showing gradual weathering/aging (the remaining images)

Evaluate "Gradual Weathering" (0-10):
Rate how natural, continuous, and gradual the progression of weathering is across the frames.
Focus on whether the frames interpolate the input image and last frame naturally.

(
    0 = The change is abrupt, flickering, or not gradual at all (e.g., sudden jumps).
    5 = The progression is somewhat gradual but has noticeable jumps or inconsistencies.
    10 = The weathering progresses perfectly smoothly and gradually from the first clean frame to the last weathered frame.
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
    0 = The structure and background are completely destroyed; the object is unrecognizable or the scene has changed entirely.
    5 = The overall shape is recognizable but there are noticeable distortions, artifacts, or unwanted changes to the background.
    10 = The structure, spatial layout, and background are perfectly preserved; only the surface appearance changes due to weathering.
)
"""
}


class MistralVIEJudge:

    def __init__(self, model_id: str = "mistral_models/Ministral-3-14B-Instruct-2512"):
        self.tokenizer = MistralCommonBackend.from_pretrained(model_id)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True)
        )
        self.device = next(self.model.parameters()).device

    def pil_to_data_url(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _gen(self, images: List[Image.Image], prompt: str) -> str:
        # Mistral expects image URLs or base64 data URLs
        content_list = []
        for img in images:
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": self.pil_to_data_url(img)
                }
            })
        
        content_list.append({
            "type": "text",
            "text": prompt
        })

        messages = [{
            "role": "user",
            "content": content_list
        }]

        tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
        
        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # Mistral needs image sizes for pixel_values
        if "pixel_values" in tokenized:
             # Assuming all input images are same size (H, W) in batch due to resizing before call
             # tokenized["pixel_values"] shape is (N, C, H, W)
             N = tokenized["pixel_values"].shape[0]
             H, W = tokenized["pixel_values"].shape[-2:]
             image_sizes = [(H, W) for _ in range(N)]
             
             output = self.model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=512,
            )
        else:
            output = self.model.generate(
                **tokenized,
                max_new_tokens=512,
            )

        output_text = self.tokenizer.decode(output[0][len(tokenized["input_ids"][0]):], skip_special_tokens=True)
        return output_text

    def evaluate_metric(self, images: List[Image.Image], output_prompt: str, metric_name: str) -> str:
        """指定されたメトリクスで評価を行う"""
        if metric_name not in _prompts:
             raise ValueError(f"Unknown metric: {metric_name}")

        # Select specific prompt
        specific_rules = _prompts[metric_name]
        
        if output_prompt:
             full_prompt = _common_context + specific_rules + f'\nTARGET WEATHERING DESCRIPTION: {output_prompt}'
        else:
             full_prompt = _common_context + specific_rules
        
        output_text = self._gen(images, full_prompt)
        return output_text


# ---------------------------
# Main evaluation
# ---------------------------
def evaluate_weathering_vie(
    json_path: str,
    gif_dir: str,
    models: List[str],
    output_dir: str = None,
    model_id: str = "mistralai/Ministral-3-14B-Instruct-2512",
    skip_frames: bool = False,
    gpu_number: int = 0,
    num_gpus: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    経年変化シミュレーションの評価 (Mistral版)
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

    judge = MistralVIEJudge(model_id=model_id)
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
            "avg_weathering_naturalness": 0.0,
            "avg_gradual_weathering": 0.0,
            "avg_structure_preservation": 0.0,
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

            output_prompt = sample["output_prompt"]

            # Evaluate each metric separately
            metrics_to_eval = ["weathering_naturalness", "gradual_weathering", "structure_preservation"]
            sample_scores = {}
            sample_reasonings = {}

            for metric in metrics_to_eval:
                try:
                    # Determine inputs based on metric
                    if metric == "weathering_naturalness":
                        # Original + Final frame, Include Prompt
                        input_images = [original_image, frames[-1]]
                        prompt_to_pass = ""
                    elif metric == "gradual_weathering":
                        # Original + All frames, No Prompt
                        input_images = [original_image] + frames
                        prompt_to_pass = ""
                    elif metric == "structure_preservation":
                         # Original + Final frame, No Prompt
                         input_images = [original_image, frames[-1]]
                         prompt_to_pass = ""
                    else:
                        continue
                    
                    # Convert input images to list for Mistral logic consistency if needed
                    # (MistralVIEJudge handles list of images)
                    
                    out_text = judge.evaluate_metric(input_images, prompt_to_pass, metric)
                    parsed = parse_json_output(out_text)
                    
                    score = parsed["score"]
                    if isinstance(score, list): 
                        score = score[0] 
                    
                    sample_scores[metric] = float(score)
                    sample_reasonings[metric] = parsed["reasoning"]
                
                except Exception as e:
                    print(f"  Error evaluating {metric} at {gif_path}: {e}")
                    print(f"  Output was: {out_text if 'out_text' in locals() else 'None'}")
                    sample_scores[metric] = 0.0
                    sample_reasonings[metric] = f"Error: {str(e)}"

            results["samples"].append({
                "image_path": sample["image_path"],
                "output_prompt": output_prompt,
                "weathering_naturalness": sample_scores["weathering_naturalness"],
                "gradual_weathering": sample_scores["gradual_weathering"],
                "structure_preservation": sample_scores["structure_preservation"],
                "reasoning": { # Store reasonings as a dict
                    "naturalness": sample_reasonings["weathering_naturalness"],
                    "gradual": sample_reasonings["gradual_weathering"],
                    "structure": sample_reasonings["structure_preservation"]
                }
            })

        # 平均を計算
        num_samples = len(results["samples"])
        if num_samples > 0:
            results["num_samples"] = num_samples
            results["avg_weathering_naturalness"] = sum(
                s["weathering_naturalness"] for s in results["samples"]
            ) / num_samples
            results["avg_gradual_weathering"] = sum(
                s["gradual_weathering"] for s in results["samples"]
            ) / num_samples
            results["avg_structure_preservation"] = sum(
                s["structure_preservation"] for s in results["samples"]
            ) / num_samples

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
                    "avg_weathering_naturalness": 0.0,
                    "avg_gradual_weathering": 0.0,
                    "avg_structure_preservation": 0.0,
                    "samples": merged_samples,
                }
                
                if final_results["num_samples"] > 0:
                     final_results["avg_weathering_naturalness"] = sum(s["weathering_naturalness"] for s in merged_samples) / final_results["num_samples"]
                     final_results["avg_gradual_weathering"] = sum(s["gradual_weathering"] for s in merged_samples) / final_results["num_samples"]
                     final_results["avg_structure_preservation"] = sum(s["structure_preservation"] for s in merged_samples) / final_results["num_samples"]

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
                 pass

        else:
            # Single GPU case
            out_file = output_path / f"vie_{model_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"  Saved: {out_file}")
        print(f"  Num samples: {results['num_samples']}")
        print(f"  Avg Weathering Naturalness: {results['avg_weathering_naturalness']:.4f}")
        print(f"  Avg Gradual Weathering: {results['avg_gradual_weathering']:.4f}")
        print(f"  Avg Structure Preservation: {results['avg_structure_preservation']:.4f}")

        all_results[model_name] = results

    return all_results


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Mistral-3 VIEScore evaluation for weathering simulation"
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
    parser.add_argument("--model_id", type=str, default="mistral_models/Ministral-3-14B-Instruct-2512",
                        help="VLM model ID or path")
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
