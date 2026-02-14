import os, json, argparse, re
from typing import List, Tuple, Dict, Any
from PIL import Image

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ---------------------------
# Config
# ---------------------------
BASE_DIR = "/work/DDIPM/okamoto_s/VideoCrafter/output"
INPUT_DIRS = []

# ---------------------------
# IO utils
# ---------------------------
def read_prompt_file(path: str, delimiter: str) -> List[Tuple[str, List[str]]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().strip('"') for p in line.split(delimiter)]
            if len(parts) < 2:
                raise ValueError(f"Each line must contain an image_path and at least one prompt, got: {line}")
            image_path, prompts = parts[0], parts[1:]
            if any(len(p) == 0 for p in prompts):
                raise ValueError(f"Empty prompt detected in line: {line}")
            items.append((image_path, prompts))
    return items

def split_strip_into_frames(strip_path: str, num_frames: int) -> List[Image.Image]:
    if not os.path.exists(strip_path):
        raise FileNotFoundError(strip_path)
    im = Image.open(strip_path).convert("RGB")
    W, H = im.size
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    tile_w = W // num_frames
    if tile_w * num_frames != W:
        raise ValueError(f"Image width {W} is not divisible by num_frames {num_frames}")
    return [im.crop((t * tile_w, 0, (t + 1) * tile_w, H)) for t in range(num_frames)]

def parse_json_output(output_text: str) -> Dict[str, Any]:
    try:
        json_start = output_text.index("{")
        json_str = output_text[json_start:]
        loaded_json = json.loads(json_str)

        if "score" not in loaded_json or not isinstance(loaded_json["score"], list):
            raise ValueError("Missing or invalid 'score' field in JSON output")

        if "reasoning" not in loaded_json:
            raise ValueError("Missing or invalid 'reasoning' field in JSON output")

        return loaded_json
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse JSON from output: {output_text}") from e

# ---------------------------
# Qwen-VIEScore evaluator
# ---------------------------

_context = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}
"""

_clarity = """
RULES:
The image frames are AI-generated according to the step-by-step instructional prompt.
The objective is to evaluate how successfully the instructional step has been depicted in the image frames.

From scale 0 to 10:
A score from 0 to 10 will rate the step faithfulness.
(
    0 indicates that the image frames do not depict their corresponding instructional step at all.
    10 indicates that each image perfectly depicts its corresponding instructional step.
)
A second score from 0 to 10 will rate the cross-image consistency.
(
    0 indicates that the image frames are completely inconsistent with each other.
    10 indicates that the image frames are perfectly consistent with each other.
)
Put the score in a list such that output score = [faithfulness, consistency]
"""

_quality = """
RULES:
The image frames are AI-generated.
The objective is to evaluate how successfully the image frames has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on the image frames naturalness.
(
    0 indicates that the scene in the image frames does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting.
    10 indicates that the image frames looks natural.
)
A second score from 0 to 10 will rate the image frames technical artifacts.
(
    0 indicates that the image frames contains a large portion of distortion, or watermark, or spurious subtitles, or noisy blurs.
    10 indicates the image frames has no technical artifacts at all.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

class QwenVIEJudge:

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        )
        self.device = next(self.model.parameters()).device

    def _gen(self, images: List[Image.Image], prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + [{"type":"text","text": prompt}]
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

    def clarity(self, images: List[Image.Image], prompt: str) -> Dict[str, Any]:
        full_prompt = _context + _clarity + f'\nINSTRUCTION: {prompt}'
        output_text = self._gen(images, full_prompt)
        return output_text

    def quality(self, images: List[Image.Image]) -> Dict[str, Any]:
        full_prompt = _context + _quality
        output_text = self._gen(images, full_prompt)
        return output_text

# ---------------------------
# Main evaluation
# ---------------------------
def evaluate_qwen_vie(
    dataset: str,
    delimiter: str,
    input_dirs: List[str],
    strip_name_template: str = "seq_{idx}.jpg",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> Dict[str, Any]:

    prompt_file = os.path.join(BASE_DIR, f"{dataset}.txt")
    lines = read_prompt_file(prompt_file, delimiter)
    judge = QwenVIEJudge(model_id=model_id)

    for input_dir in input_dirs:
        results = {
            "num_seq": 0,
            "faithfulness": 0.0,
            "consistency": 0.0,
            "SF-CIC": 0.0,
            "naturalness": 0.0,
            "artifacts": 0.0,
            "Aesthetic": 0.0,
            "seqs": [],
        }

        for idx, (_, prompts) in enumerate(tqdm(lines, desc=f"[Qwen-VIE] {input_dir}")):
            strip_path = os.path.join(BASE_DIR, input_dir, dataset, strip_name_template.format(idx=idx))
            frames = split_strip_into_frames(strip_path, num_frames=len(prompts))

            # List[str] to str
            prompt_text = "\n".join([f"Step {i+1}: {p}" for i, p in enumerate(prompts)])

            # Clarity evaluation
            clarity_out = judge.clarity(frames, prompt_text)
            try:
                clarity_json = parse_json_output(clarity_out)
                faithfulness, consistency = clarity_json["score"]
            except Exception as e:
                print(f"[Qwen-VIE] Clarity parsing error at {strip_path}: {e}")
                print(f"Output was: {clarity_out}")
                continue

            # Quality evaluation
            quality_out = judge.quality(frames)
            try:
                quality_json = parse_json_output(quality_out)
                naturalness, artifacts = quality_json["score"]
            except Exception as e:
                print(f"[Qwen-VIE] Quality parsing error at {strip_path}: {e}")
                print(f"Output was: {quality_out}")
                continue

            # Accumulate
            results["faithfulness"] += faithfulness
            results["consistency"] += consistency
            results["SF-CIC"] += min(faithfulness, consistency)
            results["naturalness"] += naturalness
            results["artifacts"] += artifacts
            results["Aesthetic"] += min(naturalness, artifacts)
            results["seqs"].append({
                "idx": idx,
                "faithfulness": faithfulness,
                "consistency": consistency,
                "naturalness": naturalness,
                "artifacts": artifacts,
                "SF-CIC": min(faithfulness, consistency),
                "Aesthetic": min(naturalness, artifacts),
                "clarity_reasoning": clarity_json["reasoning"],
                "quality_reasoning": quality_json["reasoning"],
            })

        # Average
        num_seqs = len(results["seqs"])
        results["num_seq"] = num_seqs
        results["faithfulness"] /= num_seqs
        results["consistency"] /= num_seqs
        results["SF-CIC"] /= num_seqs
        results["naturalness"] /= num_seqs
        results["artifacts"] /= num_seqs
        results["Aesthetic"] /= num_seqs

        # Save JSON
        out_path = os.path.join(BASE_DIR, input_dir, f"vie_{dataset}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"[Qwen-VIE] Saved: {out_path}")
        print(f"  Faithfulness: {results['faithfulness']:.4f}")
        print(f"  Consistency: {results['consistency']:.4f}")
        print(f"  SF-CIC: {results['SF-CIC']:.4f}")
        print(f"  Naturalness: {results['naturalness']:.4f}")
        print(f"  Artifacts: {results['artifacts']:.4f}")
        print(f"  Aesthetic: {results['Aesthetic']:.4f}")

    return results

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate sequences with Qwen3-VL using VIEScore-style metrics.")
    parser.add_argument("--dataset", type=str, choices=["showhow","wikihow"], required=True)
    parser.add_argument("--input_dirs", type=str, nargs="+", default=INPUT_DIRS) # dir1 dir2 ...
    # do not change below
    parser.add_argument("--delimiter", type=str, default="|")
    parser.add_argument("--strip_name_template", type=str, default="seq_{idx}.jpg")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    args = parser.parse_args()
    _ = evaluate_qwen_vie(
        dataset=args.dataset,
        delimiter=args.delimiter,
        input_dirs=args.input_dirs,
        strip_name_template=args.strip_name_template,
        model_id=args.model_id,
    )
    print("[Qwen-VIE] Evaluation completed.")

if __name__ == "__main__":
    main()