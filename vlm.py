import json
import os
from typing import Callable, Iterable, List, Optional, Union, Dict, Any
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


def vlm_inference(mode: str = "age", image_path: str = None):
    """
    VLMによる記述抽出
    """


    messages_age = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": "Write a brief caption describing the image, and another brief caption describing the same image in a fully deteriorated, severely weathered, or completely decayed state. Then write a simple instruction to deteriorate the image described in the first caption to its aged state described in the second caption. Do not include color information and textual information. You must use just two '|'s. Here are good examples: 'A sleek car. | A heavily rusted and moss-covered car.' | 'Add rust and moss to the car.'"
                },
            ],
        }
    ]

    messages_new = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": "Write a brief caption to describe this image in its aged state, and another to describe its predicted original clean state. And write a simple instruction restoring the image from two captions. Do not include color information, textual information. Separate them with '|'. You must use just two '|'.  Here is good example: A heavily rusted car. | A pristine clean car. | Remove rust from the car."
                }, 
            ],
        }
    ]

    if mode == "age":
        messages = messages_age
    else:
        messages = messages_new

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")


    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    orig_caption, edited_caption, instruction = output_text[0].split("|")

    return orig_caption.strip(), edited_caption.strip(), instruction.strip()


def build_prompts_json(
    image_dir: str,
    json_out: str,
    image_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"),
    sort_images: bool = True,
) -> str:
    """画像ディレクトリ内の画像に対して、VLMを用いて元画像と編集画像のキャプションを生成"""

    # 出力先ディレクトリ作成
    out_dir = os.path.dirname(json_out)
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
    print(len(images))
    records: List[Dict[str, Any]] = []
    for p in tqdm(images):
        abs_path = os.path.abspath(p)
        try:
            orig_caption, edited_caption, instruction = vlm_inference(mode="age", image_path=abs_path)
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

    return json_out

