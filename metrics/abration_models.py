import json
import argparse
from pathlib import Path
from typing import List, Optional
import sys
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import (
    FluxKontextPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    QwenImageEditPipeline,
    EulerAncestralDiscreteScheduler,
)
sys.path.append("./")
from weathering_model import WeatheringModel
sys.path.append("./abration_models")
from alltrain import AllTrainModel
from nocontrol import NoControlNetModel
from notrain import NoTrainModel
from linear import LinearModel
from sd3 import SD3Model
from alltrain_control import AllTrainControlNetModel
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
            print("proposed Model selected.")

        elif model_name == "alltrain":
            print("All Train Model selected.")

        elif model_name == "nocontrol":
            print("No ControlNet Model selected.")

        elif model_name == "notrain":
            print("No Train Model selected.")

        elif model_name == "linear":
            print("Linear Model selected.")

        elif model_name == "sd3":
            print("SD3 Model selected.")

        elif model_name == "alltrain_control":
            print("All Train ControlNet Model selected.")
            
        
        self.current_model = model_name
    
    def _unload_model(self):
        """Unload current model and release GPU memory."""
        if self.pipeline is not None:
            print(f"Unloading {self.current_model} and releasing memory...")
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Memory released.")

    
    def process_proposed(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with Proposed Model, varying guidance scale from 1 to 10."""
        self.pipeline = WeatheringModel(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames
    
    def process_alltrain(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with All Train Model, varying guidance scale from 1 to 10."""
        self.pipeline = AllTrainModel(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames
    
    def process_nocontrol(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with No ControlNet Model, varying guidance scale from 1 to 10."""
        self.pipeline = NoControlNetModel(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames
    
    def process_notrain(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with No Train Model, varying guidance scale from 1 to 10."""
        self.pipeline = NoTrainModel(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames
    
    def process_linear(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with Linear Model, varying guidance scale from 1 to 10."""
        self.pipeline = LinearModel(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames
    
    def process_sd3(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with SD3 Model, varying guidance scale from 1 to 10."""
        self.pipeline = SD3Model(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
        return frames

    def process_alltrain_control(self, image: Image.Image, input_prompt: str, output_prompt: str, num_frames: int = 10) -> List[Image.Image]:
        """Process with All Train ControlNet Model, varying guidance scale from 1 to 10."""
        self.pipeline = AllTrainControlNetModel(device=self.device)
        pipe = self.pipeline
        frames = pipe(input_image=image,
                        train_prompt=input_prompt, 
                        inference_prompt=output_prompt, 
                        negative_prompt="clean, new, pristine, undamaged, unweathered", # 経年変化用
                        attn_word=None,
                        guidance_scale=6.0,
                        num_frames=num_frames,
                    )
        self._unload_model()
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
        
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        # Use input image filename with .gif extension
        input_filename = Path(image_path).stem
        output_path = model_output_dir / f"{input_filename}.gif"
        
        try:
            if model_name == "proposed":
                frames = self.process_proposed(image, input_prompt, output_prompt, num_frames)
            elif model_name == "alltrain":
                frames = self.process_alltrain(image, input_prompt, output_prompt, num_frames)
            elif model_name == "nocontrol":
                frames = self.process_nocontrol(image, input_prompt, output_prompt, num_frames)
            elif model_name == "notrain":
                frames = self.process_notrain(image, input_prompt, output_prompt, num_frames)
            elif model_name == "linear":
                frames = self.process_linear(image, input_prompt, output_prompt, num_frames)
            elif model_name == "sd3":
                frames = self.process_sd3(image, input_prompt, output_prompt, num_frames)
            elif model_name == "alltrain_control":
                frames = self.process_alltrain_control(image, input_prompt, output_prompt, num_frames)
            
            
            save_gif(frames, str(output_path), fps=5)
            print(f"  Saved: {output_path}")
        
        except Exception as e:
            print(f"  Error with {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process images with multiple diffusion models")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./images_out")
    parser.add_argument("--models", type=str, nargs="+", required=True, choices=["proposed", "alltrain", "nocontrol", "notrain", "linear", "sd3", "alltrain_control"])
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

