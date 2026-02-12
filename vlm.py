import json
import os
from typing import Optional
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
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")


def _run_vlm(messages):
    """VLMを実行する内部関数"""
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
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def vlm_inference(mode: str = "age", image_path: str = None):
    """
    VLMによる記述抽出（3段階処理）
    1段階目: 画像を説明するキャプションを生成
    2段階目: 1段階目のキャプションの一部を変更（clean->rusted など）
    3段階目: 2つのキャプションから指示を生成
    """

    # ===== 1段階目: 画像キャプション生成 =====
    if mode == "age":
        caption_text = "Write a very short caption describing the object in this image. Do not include color name and textual information. Write only one sentence. Example: A clean car on a road."
    else:
        caption_text = "Write a very short caption describing the object in this image as it currently appears. Do not include color information and textual information. Do not describe the background. Write only one sentence. Example: A heavily rusted car."

    messages_caption = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": caption_text,
                },
            ],
        }
    ]

    caption = _run_vlm(messages_caption)
    print(f"[Stage 1: Caption] {caption}")

    # ===== 2段階目: キャプションの一部を変更 =====
    if mode == "age":
        modify_text = (
            f"The following caption describes a clean object: '{caption.strip()}'. "
            f"Rewrite this caption to describe the same object in a fully deteriorated, severely weathered, or completely decayed state. "
            f"Predict the specific type of deterioration. "
            f"Do not write shape changes such as cracks, breaks, crumbling, etc. "
            f"Do not include color name. Write only one sentence. "
            f"Do not change the background. "
            f"Example: 'A clean car on the road.' -> 'A heavily rusted car on the road.'"
        )
    else:
        modify_text = (
            f"The following caption describes an aged or deteriorated object: '{caption.strip()}'. "
            f"Rewrite this caption to describe the same object in its original clean, pristine state. "
            f"Do not include color name. Write only one sentence. "
            f"Example: 'A heavily rusted car.' -> 'A pristine clean car.'"
        )

    messages_modify = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": modify_text,
                },
            ],
        }
    ]

    modified_caption = _run_vlm(messages_modify)
    print(f"[Stage 2: Modified caption] {modified_caption}")

    if mode == "age":
        input_prompt = caption.strip()
        output_prompt = modified_caption.strip()
    else:
        input_prompt = caption.strip()
        output_prompt = modified_caption.strip()

    # ===== 3段階目: 2つのキャプションから指示を生成 =====
    if mode == "age":
        instruction_text = (
            f"Based on these two captions, write a brief instruction to age or weather the object. "
            f"Input: '{input_prompt}' -> Output: '{output_prompt}'. "
            f"Write only the instruction in one sentence. "
            f"Example: If the input is 'A clean car on a road.' and the output is 'A rusted car on a road.', the instruction is 'Add rust to the car.'"
        )
    else:
        instruction_text = (
            f"Based on these two captions, write a brief instruction to restore the object. "
            f"Input: '{input_prompt}' -> Output: '{output_prompt}'. "
            f"Write only the instruction in one sentence. "
            f"Example: 'Remove rust from the car.'"
        )

    messages_instruction = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction_text,
                },
            ],
        }
    ]

    instruction = _run_vlm(messages_instruction)
    print(f"[Stage 3: Instruction] {instruction}")

    return input_prompt, output_prompt, instruction.strip()


# "text": "Write a very short caption describing the clean state of the object, and another caption describing the same object in a fully deteriorated, severely weathered, or completely decayed state. Then write a simple instruction to deteriorate the object described in the first caption to its aged state described in the second caption. Write specific types of weathering. Do not write shape changes such as cracks, breaks, crumbling, etc. Do not include color names in all captions.  You must use just two '|'s. Here are examples: 'A clean car. | A heavily rusted car. | Add heavy rust to the car.', 'A pristine building. | A moss-covered building. | Add moss to the building.'"
