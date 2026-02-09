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
    VLMによる記述抽出（2段階処理）
    1回目: input_prompt と output_prompt を生成
    2回目: 上記プロンプトから instruction を生成
    """

    # ===== 1回目: プロンプト生成 =====
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
                    "text": "Write a very short caption describing the object, and another caption describing the same object in a fully deteriorated, severely weathered, or completely decayed state. Do not include color information and textual information. Predict the type of deterioration, such as rust on metal. Do not write shape changes such as cracks, breaks, crumbling, etc. You must use just one '|'. Here is an example: A clean car. | A heavily rusted car."
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
                    "text": "Write a very short caption to describe this image in its aged state, and another to describe its predicted original clean state. Do not include color information, textual information. Separate them with '|'. You must use just one '|'. Here is good example: A heavily rusted car. | A pristine clean car."
                }, 
            ],
        }
    ]

    if mode == "age":
        messages = messages_age
    else:
        messages = messages_new

    # 1回目のVLM実行
    output_text = _run_vlm(messages)
    print(f"[Prompt generation] {output_text}")
    
    parts = output_text.split("|")
    input_prompt = parts[0].strip()
    output_prompt = parts[1].strip() if len(parts) > 1 else ""

    # ===== 2回目: instruction生成 =====
    if mode == "age":
        instruction_text = f"Based on these two captions, write a brief instruction to age or weather the object. Input: '{input_prompt}' -> Output: '{output_prompt}'. Write only the instruction in one sentence. Example: If the input is 'A clean car.' and the output is 'A rusted car.', the instruction is 'Add rust to the car.'"
    else:
        instruction_text = f"Based on these two captions, write a brief instruction to restore the object. Input: '{input_prompt}' -> Output: '{output_prompt}'. Write only the instruction in one sentence. Example: 'Remove rust from the car.'"

    messages_instruction = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": instruction_text,
                },
            ],
        }
    ]

    # 2回目のVLM実行
    instruction = _run_vlm(messages_instruction)
    print(f"[Instruction generation] {instruction}")

    return input_prompt, output_prompt, instruction.strip()


# "text": "Write a very short caption describing the clean state of the object, and another caption describing the same object in a fully deteriorated, severely weathered, or completely decayed state. Then write a simple instruction to deteriorate the object described in the first caption to its aged state described in the second caption. Write specific types of weathering. Do not write shape changes such as cracks, breaks, crumbling, etc. Do not include color names in all captions.  You must use just two '|'s. Here are examples: 'A clean car. | A heavily rusted car. | Add heavy rust to the car.', 'A pristine building. | A moss-covered building. | Add moss to the building.'"
