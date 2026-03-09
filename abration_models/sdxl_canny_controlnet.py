import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler


DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_CONTROLNET = "diffusers/controlnet-canny-sdxl-1.0"


def build_canny_image(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges_3ch = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_3ch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SDXL + Canny ControlNet generation script")
    parser.add_argument("--input", default="images_test/image_004.jpg")
    parser.add_argument("--output", default="sdxl_canny_out.png", help="Path to output image")
    parser.add_argument("--prompt", default="A heavily rusted camera", help="Positive prompt")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="SDXL base model id")
    parser.add_argument("--controlnet_model", default=DEFAULT_CONTROLNET, help="ControlNet model id")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--control_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--low", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--high", type=int, default=200, help="Canny high threshold")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    input_image = Image.open(args.input).convert("RGB").resize((args.width, args.height), Image.LANCZOS)
    canny_image = build_canny_image(input_image, args.low, args.high)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        image=canny_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        generator=generator,
        height=args.height,
        width=args.width,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.images[0].save(args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
