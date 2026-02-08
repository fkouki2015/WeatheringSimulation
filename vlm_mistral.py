import json
import os
from typing import Optional
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


model_id = "mistralai/Ministral-3-14B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=FineGrainedFP8Config(dequantize=True)
)


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
                    "text": "Write a very short caption describing the object, and another very short caption describing the same object in a fully deteriorated, severely weathered, or completely decayed state. Then write a simple instruction to deteriorate the image described in the first caption to its aged state described in the second caption. You must write same object name in all captions. Do not mention shape changes such as cracks, breaks, crumbling, etc. Do not include color information and textual information. You must use just two '|'s. Here is an example: 'A sleek car. | A heavily rusted and moss-covered car. | Add heavy rust and moss to the car.'"
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

    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

    tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
    tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device="cuda")
    image_sizes = [tokenized["pixel_values"].shape[-2:]]

    output = model.generate(
        **tokenized,
        image_sizes=image_sizes,
        max_new_tokens=512,
    )[0]
    output_text = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
    print(output_text)
    
    orig_caption, edited_caption, instruction = output_text[0].split("|")

    return orig_caption.strip(), edited_caption.strip(), instruction.strip()


# "text": "Write a very short caption describing the clean state of the object, and another caption describing the same object in a fully deteriorated, severely weathered, or completely decayed state. Then write a simple instruction to deteriorate the object described in the first caption to its aged state described in the second caption. Write specific types of weathering. Do not write shape changes such as cracks, breaks, crumbling, etc. Do not include color names in all captions.  You must use just two '|'s. Here are examples: 'A clean car. | A heavily rusted car. | Add heavy rust to the car.', 'A pristine building. | A moss-covered building. | Add moss to the building.'"
