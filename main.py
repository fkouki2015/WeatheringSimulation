import os
from vlm import vlm_inference
from weathering_model import WeatheringModel
from weathering_model import save_gif

def load_image(path, resolution=(512, 512)):
    im = Image.open(path).convert("RGB")
    im_res = im.size
    im.resize(resolution, Image.LANCZOS)
    return im

def process_folder(input_folder: str, output_folder: str, train_prompt: str, inference_prompt: str):
    model = WeatheringModel(device="cuda")
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(input_folder, filename)
        input_image = load_image(image_path, resolution=(512, 512))
        if train_prompt is None or inference_prompt is None:
            train_prompt, inference_prompt= vlm_inference(mode="age", image_path=image_path)

        print(f"Processing {filename}")
        print(f"  Train Prompt: {train_prompt}")
        print(f"  Inference Prompt: {inference_prompt}")
        
        output_frames = model(
            input_image=input_image,
            train_prompt=train_prompt,
            inference_prompt=inference_prompt,
            negative_prompt="",
            attn_word="asdt",
            guidance_scale=6.0,
            num_frames=10,
        )
        
        output_path = os.path.join(output_folder, f"output_{filename.rsplit('.',1)[0]}.gif")
        save_gif(output_frames, output_path, fps=4, loop=0)
        print(f"Saved output to {output_path}")


def process_image(image_path: str, output_folder: str, train_prompt: str, inference_prompt: str):
    model = WeatheringModel(device="cuda")
    input_image = load_image(image_path, resolution=(512, 512))
    if train_prompt is None or inference_prompt is None:
        train_prompt, inference_prompt= vlm_inference(mode="age", image_path=image_path)

    print(f"  Train Prompt: {train_prompt}")
    print(f"  Inference Prompt: {inference_prompt}")
    
    output_frames = model(
        input_image=input_image,
        train_prompt=train_prompt,
        inference_prompt=inference_prompt,
        negative_prompt="",
        attn_word="asdt",
        guidance_scale=6.0,
        num_frames=10,
    )
    
    output_path = os.path.join(output_folder, f"output.gif")
    save_gif(output_frames, output_path, fps=4, loop=0)
    print(f"Saved output to {output_path}")



if __name__ == "__main__":
    process_image("images/image_01.jpg", "output", None, None)