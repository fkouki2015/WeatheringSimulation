import gc
import json
import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
from typing import Optional
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys

# モデルは遅延ロード（import vlm してもすぐには読み込まない）
_vlm_model = None
_vlm_processor = None


def _get_vlm():
    """VLMモデルを遅延ロードして返す"""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print("Loading VLM model...")
        _vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
        )
        _vlm_model.eval()
        _vlm_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct"
        )
        print("VLM model loaded.")
    return _vlm_model, _vlm_processor


def unload_vlm():
    """VLMモデルをメモリから解放する"""
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        print("Unloading VLM model...")
        del _vlm_model
        del _vlm_processor
        _vlm_model = None
        _vlm_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VLM model unloaded.")


def _run_vlm(messages):
    """VLMを実行する内部関数"""
    from qwen_vl_utils import process_vision_info
    model, processor = _get_vlm()
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
        caption_text = "Write the object or scene name in this image. Do not include color name and textual information. You must write only one word. Do not describe the background. Example: car"
    else:
        caption_text = "Write the object or scene name in this image. Do not include color information and textual information. Do not describe the background. Write only few words. Example: rusted car"

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
            f"The following caption describes a clean object or scene: '{caption.strip()}'. "
            f"Rewrite this caption to describe the same object or scene in a fully deteriorated, severely weathered, or completely decayed state. "
            f"Write the specific type of deterioration. "
            f"Do not write shape changes such as cracks, breaks, crumbling, etc. "
            f"Do not write color name. Write only a few words. "
            f"Example: 'car' -> 'heavily rusted car'"
        )
    else:
        modify_text = (
            f"The following caption describes an aged or deteriorated object: '{caption.strip()}'. "
            f"Rewrite this caption to describe the same object in its original clean, pristine state. "
            f"Do not include color name. Write only a few words. "
            f"Example: 'rusted car' -> 'clean car'"
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
            f"Example: If the input is 'car' and the output is 'rusted car', the instruction is 'Add rust to the car.'"
        )
    else:
        instruction_text = (
            f"Based on these two captions, write a brief instruction to restore the object. "
            f"Input: '{input_prompt}' -> Output: '{output_prompt}'. "
            f"Write only the instruction in one sentence. "
            f"Example: If the input is 'rusted car' and the output is 'clean car', the instruction is 'Remove rust from the car.'"
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
