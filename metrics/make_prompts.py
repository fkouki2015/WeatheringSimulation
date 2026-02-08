import json
import os
import sys
import argparse
from typing import Iterable, List, Dict, Any
from tqdm import tqdm
sys.path.append("./")
from vlm import vlm_inference


def build_prompts_json(
    image_dir: str,
    json_out: str,
    mode: str = "age",
    image_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"),
    sort_images: bool = True,
) -> str:
    """画像ディレクトリ内の画像に対して、VLMを用いて元画像と編集画像のキャプションを生成"""

    # 出力先ディレクトリ作成
    out_dir = os.path.dirname(json_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 画像列挙
    names = os.listdir(image_dir)
    exts = tuple(ext.lower() for ext in image_extensions)
    images = [
        os.path.join(image_dir, name)
        for name in names
        if os.path.isfile(os.path.join(image_dir, name)) and name.lower().endswith(exts)
    ]
    if sort_images:
        images.sort(key=lambda p: os.path.basename(p))
    
    print(f"Found {len(images)} images")
    
    records: List[Dict[str, Any]] = []
    for p in tqdm(images):
        abs_path = os.path.abspath(p)
        try:
            orig_caption, edited_caption, instruction = vlm_inference(mode=mode, image_path=abs_path)
            sys.stdout.flush()
            rec = {
                "image_path": abs_path,
                "input_prompt": orig_caption,
                "output_prompt": edited_caption,
                "edit": instruction,
            }
            records.append(rec)
        except Exception as e:
            tqdm.write(f"Error processing image {abs_path}: {e}")
            sys.stdout.flush()
            

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved prompts to {json_out}")
    return json_out


def main():
    parser = argparse.ArgumentParser(description="Generate prompts JSON from images using VLM")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--json_out", type=str, required=True)
    parser.add_argument("--mode", type=str, default="age", choices=["age", "restore"])
    args = parser.parse_args()
    
    build_prompts_json(
        image_dir=args.image_dir,
        json_out=args.json_out,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
