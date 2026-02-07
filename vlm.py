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
                    "text": "Write a very short caption describing the clean state of the image, and another very short caption describing the same image in a fully deteriorated state. Then write a simple instruction to deteriorate the image described in the first caption to its aged state described in the second caption. Predict the most likely type of deterioration. Limit types of deterioration to texture changes. Do not include color names.  You must use just two '|'s. Here are examples: 'A clean car. | A heavily rusted car.' | 'Add heavy rust to the car.', 'A pristine building. | A moss-covered building.' | 'Add moss to the building.'"
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
                    "text": "Write a very short caption to describe this image in its aged state, and another to describe its predicted original clean state. And write a simple instruction restoring the image from two captions. Do not include color information, textual information. Separate them with '|'. You must use just two '|'.  Here is good example: A heavily rusted car. | A pristine clean car. | Remove rust from the car."
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

