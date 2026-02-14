import os
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from diffusers import AutoPipelineForImage2Image, FluxPipeline


class FluxModel(nn.Module):
    RESOLUTION = (1024, 1024)
    PRETRAINED_MODEL = "black-forest-labs/FLUX.1-dev"
    DEVICE = "cuda"
    INFER_STEPS = 28
    MAX_SEQUENCE_LENGTH = 256

    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.pipeline = None
        self.uses_img2img = False
        self._init_pipeline()

    def _init_pipeline(self):
        # Keep loading strict-local for offline execution.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                self.PRETRAINED_MODEL,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
            self.uses_img2img = True
        except Exception:
            self.pipeline = FluxPipeline.from_pretrained(
                self.PRETRAINED_MODEL,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
            self.uses_img2img = False

        self.pipeline.to(self.device)
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

    def _frame_prompt(self, base_prompt: str, ratio: float) -> str:
        intensity = int(round(ratio * 100))
        return (
            f"{base_prompt}. "
            f"Weathering intensity {intensity}% while preserving object identity and composition."
        )

    def _effective_guidance(self, guidance_scale: float) -> float:
        # FLUX.1-dev is typically more stable around lower guidance than SD3.x defaults.
        return float(max(1.0, min(guidance_scale, 4.0)))

    def forward(
        self,
        input_image: Image.Image,
        train_prompt: str,
        inference_prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_frames: int,
    ) -> List[Image.Image]:
        del train_prompt
        del negative_prompt

        if num_frames <= 0:
            return []

        input_image = input_image.convert("RGB").resize(self.RESOLUTION, Image.LANCZOS)
        frames: List[Image.Image] = []
        gscale = self._effective_guidance(guidance_scale)

        for i in range(num_frames):
            ratio = i / max(1, num_frames - 1)
            prompt = self._frame_prompt(inference_prompt, ratio)
            generator = torch.Generator(device="cpu").manual_seed(1234 + i)

            if self.uses_img2img:
                strength = 0.15 + 0.8 * ratio
                out = self.pipeline(
                    prompt=prompt,
                    image=input_image,
                    strength=float(max(0.0, min(1.0, strength))),
                    guidance_scale=gscale,
                    num_inference_steps=self.INFER_STEPS,
                    generator=generator,
                    max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                ).images[0]
            else:
                out = self.pipeline(
                    prompt=prompt,
                    height=self.RESOLUTION[1],
                    width=self.RESOLUTION[0],
                    guidance_scale=gscale,
                    num_inference_steps=self.INFER_STEPS,
                    generator=generator,
                    max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                ).images[0]

            frames.append(out.resize((512, 512), Image.LANCZOS))

        return frames


# Backward compatibility for code paths that still import SD3Model from flux.py.
SD3Model = FluxModel
