import json
import argparse
from pathlib import Path
from typing import List, Optional
import sys
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.append("./")
sys.path.append("/work/DDIPM/kfukushima/FlowEdit")

from weathering_model import WeatheringModel
from FlowEdit_utils import FlowEditFLUX
sys.path.append("/work/DDIPM/kfukushima/turbo-edit")
from turbo import encode_image, set_pipeline, run, load_pipe
import gc


def save_gif(frames: List[Image.Image], out_path: str, fps: int = 5, loop: int = 0):
    """Save a list of PIL images as a GIF file."""
    if not frames:
        return
    
    base = frames[0]
    size = base.size
    proc = []
    for im in frames:
        if im.size != size:
            im = im.resize(size, Image.LANCZOS)
        if im.mode != "P":
            im = im.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT)
        proc.append(im)
    duration = max(1, int(1000 / max(1, fps)))
    proc[0].save(
        out_path,
        save_all=True,
        append_images=proc[1:],
        duration=duration,
        loop=loop,
        optimize=True,
        disposal=2,
    )


class ModelProcessor:
    
    def __init__(self, models: List[str], device: str = "cuda"):
        self.device = device
        self.models = models
        self.pipeline = None  # Single pipeline at a time
        self.current_model = None
    
    def _load_model(self, model_name: str):
        """Load a single model pipeline."""
        if model_name == "proposed":
            print("Loading Proposed Model...")

        elif model_name == "flux":
            print("Loading Flux Kontext...")
            from diffusers import FluxKontextPipeline
            self.pipeline = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).to(self.device)

        elif model_name == "qwen":
            print("Loading Qwen Image Edit...")
            from diffusers import QwenImageEditPipeline
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit", 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).to(self.device)

        elif model_name == "ip2p":
            print("Loading InstructPix2Pix...")
            from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix", 
                torch_dtype=torch.float16, 
                safety_checker=None,
                local_files_only=True
            ).to(self.device)
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)

        elif model_name == "sd":
            print("Loading Stable Diffusion...")
            from diffusers import StableDiffusionPipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16, 
                safety_checker=None,
                local_files_only=True
            ).to(self.device)

        elif model_name == "sdedit":
            print("Loading SDEdit (Img2Img)...")
            from diffusers import StableDiffusionImg2ImgPipeline
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16, 
                safety_checker=None,
                local_files_only=True
            ).to(self.device)

        elif model_name == "flowedit":
            print("Loading FlowEdit (FLUX)...")
            from diffusers import FluxPipeline
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16,
                local_files_only=True
            ).to(self.device)

        elif model_name == "turboedit":
            print("Loading TurboEdit...")
            self.pipeline = load_pipe(False, None)
        
        self.current_model = model_name
    
    def _unload_model(self):
        """Unload current model and release GPU memory."""
        if self.pipeline is not None:
            print(f"Unloading {self.current_model} and releasing memory...")
            del self.pipeline
            self.pipeline = None
            self.current_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Memory released.")

    
    def process_proposed(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with Proposed Model, varying guidance scale from 1 to 10."""
        self.pipeline = WeatheringModel(device=self.device)
        self.current_model = "proposed"
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        # negative_prompt="",
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames
    
    def process_flux(self, image: Image.Image, prompt: str, num_frames: int = 10, height: int = 512, width: int = 512) -> List[Image.Image]:
        """Process with Flux Kontext, varying guidance scale from 1 to 10."""
        pipe = self.pipeline
        frames = []
        for scale in range(1, num_frames + 1):
            torch.manual_seed(0)
            result = pipe(
                image=image,
                prompt=prompt,
                guidance_scale=float(scale),
                num_inference_steps=28,
            ).images[0].resize((width, height), resample=Image.LANCZOS)
            frames.append(result)
        return frames
    
    def process_qwen(self, image: Image.Image, prompt: str, num_frames: int = 10, height: int = 512, width: int = 512) -> List[Image.Image]:
        """Process with Qwen Image Edit, varying guidance scale from 1 to 10."""
        pipe = self.pipeline
        frames = []
        for scale in range(1, num_frames + 1):
            inputs = {
                    "image": image,
                    "prompt": prompt,
                    "generator": torch.manual_seed(0),
                    "true_cfg_scale": scale,
                    "negative_prompt": " ",
                    "num_inference_steps": 50,
                }
            with torch.inference_mode():
                frames.append(pipe(**inputs).images[0].resize((width, height), resample=Image.LANCZOS))
        return frames
    
    def process_ip2p(self, image: Image.Image, prompt: str, num_frames: int = 10, height: int = 512, width: int = 512) -> List[Image.Image]:
        """Process with InstructPix2Pix, varying guidance scale from 1 to 10."""
        pipe = self.pipeline
        frames = []
        for scale in range(1, num_frames + 1):
            torch.manual_seed(0)
            result = pipe(
                image=image,
                prompt=prompt,
                guidance_scale=float(scale),
                num_inference_steps=50,
            ).images[0].resize((width, height), resample=Image.LANCZOS)
            frames.append(result)
        return frames
    
    def process_sd(self, prompt: str, num_frames: int = 10, height: int = 512, width: int = 512) -> List[Image.Image]:
        """Process with Stable Diffusion (text-to-image), varying guidance scale from 1 to 10."""
        pipe = self.pipeline
        frames = []
        for scale in range(1, num_frames + 1):
            torch.manual_seed(0)
            result = pipe(
                prompt=prompt,
                guidance_scale=float(scale),
                num_inference_steps=50,
                height=height,
                width=width,
            ).images[0]
            frames.append(result)
        return frames
    
    def process_sdedit(self, image: Image.Image, prompt: str, num_frames: int = 10, height: int = 512, width: int = 512) -> List[Image.Image]:
        """Process with SDEdit (Img2Img), varying strength from 0.1 to 1.0."""
        pipe = self.pipeline
        frames = []
        for i in range(1, num_frames + 1):
            torch.manual_seed(0)
            strength = i / num_frames  # 0.1, 0.2, ..., 1.0
            result = pipe(
                image=image,
                prompt=prompt,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=50,
            ).images[0].resize((width, height), resample=Image.LANCZOS)
            frames.append(result)
        return frames
    
    def process_flowedit(self, image: Image.Image, src_prompt: str, tar_prompt: str, num_frames: int = 10, height: int = 512, width: int = 512) -> List[Image.Image]:
        """Process with FlowEdit (FLUX), varying tar_guidance_scale from 1.0 to 10.0."""
        pipe = self.pipeline
        scheduler = pipe.scheduler
        
        # Encode source image to latent
        image_cropped = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
        image_src = pipe.image_processor.preprocess(image_cropped)
        image_src = image_src.to(self.device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
        x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        x0_src = x0_src.to(self.device)
        
        frames = []
        torch.manual_seed(0)
        np.random.seed(0)
        
        x0_tar, intermediates = FlowEditFLUX(
            pipe,
            scheduler,
            x0_src,
            src_prompt,
            tar_prompt,
            negative_prompt="",
            T_steps=28,
            n_avg=1,
            src_guidance_scale=1.5,
            tar_guidance_scale=5.5,
            n_min=0,
            n_max=24,
            return_intermediates=True
        )

        for step_idx, latent in enumerate(intermediates):
            if step_idx % 3 != 2:
                continue
            latent = latent.to(self.device)
            x_denorm = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                img = pipe.vae.decode(x_denorm, return_dict=False)[0]
            img = pipe.image_processor.postprocess(img)
            frames.append(img[0].resize((512, 512), resample=Image.LANCZOS))
        return frames

    def process_turboedit(self, image: Image.Image, src_prompt: str, tar_prompt: str, num_frames: int, height: int=512, width: int=512) -> List[Image.Image]:
        """Process with TurboEdit, varying tar_guidance_scale from 1.0 to 10.0."""
        pipe = self.pipeline
        frames = []
        for i in range(1, num_frames + 1):
            strength = i / 6.0  # 0.1, 0.2, ..., 1.0
            res = run(
                image,
                src_prompt,
                tar_prompt,
                0,
                strength,
                4,
                pipeline=pipe,
            )
            frames.append(res)
        return frames
    
    def process_sample(
        self, 
        sample: dict, 
        output_dir: Path, 
        sample_idx: int, 
        model_name: str,
        num_frames: int = 10
    ):
        """Process a single sample with the currently loaded model."""
        image_path = sample["image_path"]
        input_prompt = sample["input_prompt"]
        output_prompt = sample["output_prompt"]
        edit_prompt = sample["edit"]
        # edit_prompt = ""
        
        image = Image.open(image_path).convert("RGB")
        # Resize to 512x512
        image = image.resize((512, 512), Image.LANCZOS)
        image_large = image.resize((1024, 1024), Image.LANCZOS)
        
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        # Use input image filename with .gif extension
        input_filename = Path(image_path).stem
        output_path = model_output_dir / f"{input_filename}.gif"
        
        try:
            if model_name == "proposed":
                frames = self.process_proposed(image, input_prompt, output_prompt, num_frames)
            elif model_name == "flux":
                frames = self.process_flux(image_large, edit_prompt, num_frames)
            elif model_name == "qwen":
                frames = self.process_qwen(image_large, edit_prompt, num_frames)
            elif model_name == "ip2p":
                frames = self.process_ip2p(image, edit_prompt, num_frames)
            elif model_name == "sd":
                h, w = image.size
                frames = self.process_sd(output_prompt, num_frames, h, w)
            elif model_name == "sdedit":
                frames = self.process_sdedit(image, output_prompt, num_frames)
            elif model_name == "flowedit":
                frames = self.process_flowedit(image_large, input_prompt, output_prompt, num_frames)
            elif model_name == "turboedit":
                frames = self.process_turboedit(image, input_prompt, output_prompt, num_frames)
            save_gif(frames, str(output_path), fps=5)
            print(f"  Saved: {output_path}")
        
        except Exception as e:
            print(f"  Error with {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process images with multiple diffusion models")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./images_out")
    parser.add_argument("--models", type=str, nargs="+", required=True, choices=["proposed", "flux", "qwen", "ip2p", "sd", "sdedit", "flowedit", "turboedit"])
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_number", type=int, default=0, help="Current GPU number (0 ~ num_gpus-1)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs for parallel processing")
    args = parser.parse_args()
    # Load JSON data
    with open(args.json_path, "r") as f:
        data = json.load(f)
    
    # Handle both list and single dict formats
    if isinstance(data, dict):
        data = [data]
    
    # Filter samples for this GPU (round-robin assignment)
    my_samples = [(idx, sample) for idx, sample in enumerate(data) if idx % args.num_gpus == args.gpu_number]
    
    print(f"GPU {args.gpu_number}/{args.num_gpus}: Processing {len(my_samples)}/{len(data)} samples")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor (no models loaded yet)
    processor = ModelProcessor(args.models, args.device)
    
    # Process one model at a time: load -> process all images -> unload
    for model_name in args.models:
        print(f"Processing with model: {model_name}")
        
        # Load the model
        processor._load_model(model_name)
        
        # Process all assigned samples with this model
        for idx, sample in tqdm(my_samples, desc=f"{model_name}"):
            print(f"\nProcessing sample {idx}: {sample.get('image_path', 'unknown')}")
            processor.process_sample(sample, output_dir, idx, model_name, args.num_frames)
        
        # Unload the model and release memory
        processor._unload_model()
    
    print(f"\nGPU {args.gpu_number} done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

