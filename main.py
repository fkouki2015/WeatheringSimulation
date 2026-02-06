import os
import argparse
from PIL import Image
from vlm import vlm_inference
from weathering_model import WeatheringModel

def load_image(path, resolution=(512, 512)):
    im = Image.open(path).convert("RGB")
    im_res = im.size
    im = im.resize(resolution, Image.LANCZOS)
    return im

def save_gif(frames, out_path, fps=12, loop=0): 
    """PIL画像のリストをGIFとして保存する。"""
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
    duration = max(1, int(1000 / max(1, fps)))  # ミリ秒/フレーム
    proc[0].save(
        out_path,
        save_all=True,
        append_images=proc[1:],
        duration=duration,
        loop=loop,
        optimize=True,
        disposal=2,
    )

def process_folder(input_folder: str, output_folder: str, mode: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = WeatheringModel(device="cuda")
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(input_folder, filename)
        input_image = load_image(image_path, resolution=(512, 512))
        
        train_prompt, inference_prompt, instruction = vlm_inference(mode=mode, image_path=image_path)

        print(f"処理中 {filename}")
        print(f"  訓練用プロンプト: {train_prompt}")
        print(f"  推論用プロンプト: {inference_prompt}")
        
        output_frames = model(
            input_image=input_image,
            train_prompt=train_prompt,
            inference_prompt=inference_prompt,
            negative_prompt="",
            guidance_scale=6.0,
            num_frames=10,
        )
        
        output_path = os.path.join(output_folder, f"output_{filename.rsplit('.',1)[0]}.gif")
        save_gif(output_frames, output_path, fps=4, loop=0)
        print(f"Saved output to {output_path}")




def process_image(image_path: str, output_folder: str, mode: str, train_prompt: str, inference_prompt: str, attn_word: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = WeatheringModel(device="cuda")
    input_image = load_image(image_path, resolution=(512, 512))
    
    if train_prompt is None or inference_prompt is None:
        train_prompt, inference_prompt, instruction = vlm_inference(mode=mode, image_path=image_path)

    print(f"処理中 {image_path}")
    print(f"  訓練用プロンプト: {train_prompt}")
    print(f"  推論用プロンプト: {inference_prompt}")
    if attn_word is not None:
        print(f"  強調単語: {attn_word}")
    
    output_frames = model(
        input_image=input_image,
        train_prompt=train_prompt,
        inference_prompt=inference_prompt,
        negative_prompt="clean, new",
        attn_word=attn_word,
        guidance_scale=6.0,
        num_frames=5,
    )
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"output_{filename.rsplit('.',1)[0]}.gif")
    save_gif(output_frames, output_path, fps=4, loop=0)
    print(f"保存先 {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="経年変化シミュレーション")
    parser.add_argument("--input_folder", type=str, help="一括処理するフォルダ")
    parser.add_argument("--input_image", type=str, help="単一画像の処理")
    parser.add_argument("--output_folder", type=str, default="outputs", help="出力先フォルダ")
    parser.add_argument("--mode", type=str, default="age", choices=["age", "restore"], help="経年変化 or 復元")
    parser.add_argument("--train_prompt", type=str, help="訓練用のキャプション（オプション）")
    parser.add_argument("--inference_prompt", type=str, help="推論用のキャプション（オプション）")
    parser.add_argument("--attn_word", type=str, default=None, help="強調する単語（オプション）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.input_folder:
        process_folder(
            args.input_folder, 
            args.output_folder, 
            args.mode,
            args.train_prompt, 
            args.inference_prompt, 
            args.attn_word
        )
    elif args.input_image:
        process_image(
            args.input_image, 
            args.output_folder, 
            args.mode,
            args.train_prompt, 
            args.inference_prompt, 
            args.attn_word
        )
    else:
        print("--input_folderまたは--input_imageを指定してください．")


