import os
import sys
import uuid
import copy
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import lpips
import diffusers
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKL, 
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    SD3ControlNetModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ==========================================
# ユーティリティ関数
# ==========================================

def compute_perceptual_distance(pil_image1: Image.Image, pil_image2: Image.Image, lpips_model, device) -> float:
    """2つの画像間のLPIPS距離を計算"""
    if pil_image1.size != pil_image2.size:
        pil_image2 = pil_image2.resize(pil_image1.size, Image.LANCZOS)
    
    def to_normed_tensor(img: Image.Image):
        t = transforms.ToTensor()(img).unsqueeze(0).to(device)
        return t * 2.0 - 1.0  # [-1, 1]
    
    with torch.no_grad():
        d = lpips_model(to_normed_tensor(pil_image1), to_normed_tensor(pil_image2))
    return float(d.mean().item())


def canny_process(image, device, dtype):
    """画像からCannyエッジマップを生成"""
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    canny_img = cv2.Canny(gray, 100, 200)
    
    control_image = Image.fromarray(canny_img).convert("RGB")
    return control_image


def pil_to_latent(vae: AutoencoderKL, pil: Image.Image, device, dtype):
    x = transforms.ToTensor()(pil).unsqueeze(0).to(device)
    x = (x * 2 - 1).to(dtype=torch.float32)
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample()

    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.to(dtype=dtype)


def control_pil_to_latent_sd3_canny_equiv(
    vae: AutoencoderKL,
    pil: Image.Image,
    resolution,
    device,
    dtype,
):
    x = transforms.ToTensor()(pil).unsqueeze(0).to(device=device, dtype=torch.float32)
    # SD3CannyImageProcessor.preprocess と同等の変換
    x = x * 255.0 * 0.5 + 0.5
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.to(dtype=dtype)


def latent_to_pil(vae: AutoencoderKL, latents):
    """潜在ベクトルをPIL画像にデコード (SD3用)"""
    latents_decode = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    latents_decode = latents_decode.to(dtype=vae.dtype)
    with torch.no_grad():
        imgs = vae.decode(latents_decode).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = (imgs * 255).round().to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    pil_list = [Image.fromarray(arr) for arr in imgs]
    return pil_list

def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1


def _encode_prompt_clip(
    text_encoder,
    tokenizer,
    prompt_list,
    device,
    dtype,
):
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    outputs = text_encoder(text_input_ids, output_hidden_states=True)
    pooled_prompt_embeds = outputs[0].to(dtype=dtype, device=device) # shape: (batch, dim) = (1, 768) for SD3.5 large
    prompt_embeds = outputs.hidden_states[-2].to(dtype=dtype, device=device) # shape: (batch, seq_len, dim) = (1, 77, 768) for SD3.5 large

    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt_t5(
    text_encoder,
    tokenizer,
    prompt_list,
    max_sequence_length,
    device,
    dtype,
):
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    outputs = text_encoder(text_input_ids)[0]
    prompt_embeds = outputs.to(dtype=dtype, device=device) 

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt_list,
    max_sequence_length,
    device,
    dtype,
):
    clip1_embed, clip1_pooled = _encode_prompt_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt_list=prompt_list,
        device=device,
        dtype=dtype,
    )
    clip2_embed, clip2_pooled = _encode_prompt_clip(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        prompt_list=prompt_list,
        device=device,
        dtype=dtype,
    )
    t5_embed = _encode_prompt_t5(
        text_encoder=text_encoders[2],
        tokenizer=tokenizers[2],
        prompt_list=prompt_list,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )

    clip_embeds = torch.cat([clip1_embed, clip2_embed], dim=-1)  # (batch, seq_len, 1536)
    clip_embeds = torch.nn.functional.pad(clip_embeds, (0, t5_embed.shape[-1] - clip_embeds.shape[-1]))  # 最終次元から左, 右の順でパディング
    prompt_embeds = torch.cat([clip_embeds, t5_embed], dim=1)  # (batch, seq_len + t5_seq_len, 1536)
    pooled_embeds = torch.cat([clip1_pooled, clip2_pooled], dim=-1)  # (batch, 1536)
    
    return prompt_embeds, pooled_embeds


# ==========================================
# 経年変化モデル (SD3.5対応)
# ==========================================

class SD3Model(nn.Module):
    # デフォルト定数
    RESOLUTION = (1024, 1024)
    RANK = 8
    LEARNING_RATE = 1e-6
    TRAIN_STEPS = 300
    PRETRAINED_MODEL = "stabilityai/stable-diffusion-3.5-large"
    CONTROLNET_PATH = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
    DEVICE = "cuda"
    
    # 評価設定
    CLIP_EVAL_INTERVAL = 50
    CLIP_EVAL_STEPS = 20
    PERCEPTUAL_THRESHOLD = 0.05
    PERCEPTUAL_PATIENCE = 2
    
    # 推論設定
    INFER_STEPS = 40
    NOISE_RATIO = 0.6
    MAX_SEQUENCE_LENGTH = 256
    
    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.bfloat16

        # スケジューラ
        self.noise_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="scheduler", 
        )
        # トークナイザ
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer", 
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer_2", 
        )
        self.tokenizer_three = T5TokenizerFast.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer_3", 
        )
        # テキストエンコーダ
        self.text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder"
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder_2"
        )
        self.text_encoder_three = T5EncoderModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder_3"
        )
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="vae", 
        )
        # Transformer
        self.transformer = SD3Transformer2DModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="transformer", 
        )
        # ControlNet
        self.controlnet = SD3ControlNetModel.from_pretrained(
            self.CONTROLNET_PATH, torch_dtype=self.dtype
        )
        # LPIPS
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()

        # デバイスへ移動
        self.vae.to(device=self.device, dtype=torch.float32)
        self.transformer.to(device=self.device, dtype=self.dtype)
        self.text_encoder_one.to(device=self.device, dtype=self.dtype)
        self.text_encoder_two.to(device=self.device, dtype=self.dtype)
        self.text_encoder_three.to(device=self.device, dtype=self.dtype)
        self.controlnet.to(device=self.device, dtype=self.dtype)

        # デフォルトで重みを固定
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.text_encoder_three.requires_grad_(False)
        self.controlnet.requires_grad_(False)


    def _setup_training(self):
        """トレーニング用にLoRAとオプティマイザを設定"""
        transformer_lora_config = LoraConfig(
            r=self.RANK,
            lora_alpha=self.RANK,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                "attn.to_out.0", "attn.to_v", "attn.to_q", "attn.to_k"
            ]
        )
        self.transformer.add_adapter(transformer_lora_config)
        # self.transformer.requires_grad_(True)
        # self.controlnet.requires_grad_(True)
        
        groups_by_scale = {}
        
        def _add_param(param, scale: float):
            key = round(float(scale), 6)
            if key not in groups_by_scale:
                groups_by_scale[key] = []
            groups_by_scale[key].append(param)
        
        for n, p in self.transformer.named_parameters():
            if p.requires_grad:
                _add_param(p, 1.0)

        for n, p in self.controlnet.named_parameters():
            if p.requires_grad:
                _add_param(p, 1.0)
        
        param_groups = [
            {"params": param, "lr": self.LEARNING_RATE * scale}
            for scale, param in groups_by_scale.items()
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )
        
        return

    def train_model(self):
        self._setup_training()
        
        x_0 = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)

        prompt_embeds, pooled_embeds = encode_prompt(
            [self.text_encoder_one, self.text_encoder_two, self.text_encoder_three],
            [self.tokenizer_one, self.tokenizer_two, self.tokenizer_three],
            [self.train_prompt],
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
            device=self.device,
            dtype=self.dtype,
        )

        self.transformer.train()
        self.controlnet.train()

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for step in progress_bar:
            z = torch.randn_like(x_0)
            t_float = sample_t_logit_normal(batch_size=x_0.shape[0], device=x_0.device)
            t = t_float.view(-1, 1, 1, 1).to(self.dtype)
            x_t = (1.0 - t) * x_0 + t * z
            
            # ControlNet順伝播
            controlnet_block_samples = self.controlnet(
                hidden_states=self.transformer.pos_embed(x_t),
                controlnet_cond=self.control_image_tensor,
                timestep=t_float * self.noise_scheduler.num_train_timesteps,
                pooled_projections=pooled_embeds,
                conditioning_scale=1.0,
                return_dict=False,
            )[0]
            
            # Transformer順伝播
            pred_v = self.transformer(
                hidden_states=x_t,
                timestep=t_float * self.noise_scheduler.num_train_timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                block_controlnet_hidden_states=controlnet_block_samples,
                return_dict=False,
            )[0]

            pred_x = pred_v * (-t) + x_t
            loss = torch.nn.functional.mse_loss(pred_x.float(), x_0.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})

    def forward(
        self,    
        input_image: Image.Image,
        train_prompt: str,
        inference_prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_frames: int,
    ) -> list[Image.Image]:
        
        # パラメータを保存
        self.input_image = input_image.resize(self.RESOLUTION, Image.LANCZOS)
        self.train_prompt = train_prompt
        
        self.control_image = canny_process(self.input_image, self.device, self.dtype)
        self.control_image_tensor = control_pil_to_latent_sd3_canny_equiv(
            self.vae,
            self.control_image,
            self.RESOLUTION,
            self.device,
            self.dtype,
        )
        

        # 1. モデルのトレーニング
        self.train_model()
        
        # 2. 推論のセットアップ
        torch.manual_seed(1234)
        self.vae.eval()
        self.transformer.eval()
        self.controlnet.eval()
        self.text_encoder_one.eval()
        self.text_encoder_two.eval()
        self.text_encoder_three.eval()
        
        # テキスト埋め込み
        prompt_embeds, pooled_embeds = encode_prompt(
            [self.text_encoder_one, self.text_encoder_two, self.text_encoder_three],
            [self.tokenizer_one, self.tokenizer_two, self.tokenizer_three],
            [negative_prompt, inference_prompt],
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
            device=self.device,
            dtype=self.dtype,
        )
        
        # 潜在変数の準備
        x_0 = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)

        frames = []
        # 3. 生成ループ (フレームごと)
        for i in range(num_frames):
            torch.manual_seed(1234)
            self.noise_scheduler.set_timesteps(self.INFER_STEPS, device=self.device)
            timesteps = self.noise_scheduler.timesteps # 0..1000のうち、INFER_STEPS個を等間隔で選択したもの
            
            # t_index計算
            normalized_i = i / max(1, num_frames - 1)
            t_index = self.INFER_STEPS - int((self.INFER_STEPS - 1) * (normalized_i ** self.NOISE_RATIO)) - 1
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            if num_frames == 1:
                t_index = 0
            # t_index: 0...INFER_STEPS-1

            t_float = timesteps[t_index] / self.noise_scheduler.num_train_timesteps
            t = t_float.view(-1, 1, 1, 1).to(self.dtype)
            z = torch.randn_like(x_0)
            x_t = (1.0 - t) * x_0 + t * z
            
            # デノイズ
            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i+1}/{num_frames}")
            for j, index in enumerate(progress):
                with torch.no_grad():
                    timestep = timesteps[index].to(self.device) # 0..1000
                    timestep_in = timestep.repeat(2)
                
                    x_t_in = x_t.repeat(2, 1, 1, 1)
                    control_in = self.control_image_tensor.repeat(2, 1, 1, 1)
                    
                    controlnet_block_samples = self.controlnet(
                        hidden_states=self.transformer.pos_embed(x_t_in),
                        controlnet_cond=control_in,
                        timestep=timestep_in,
                        pooled_projections=pooled_embeds,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )[0]
                    
                    model_out = self.transformer(
                        hidden_states=x_t_in,
                        timestep=timestep_in,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        block_controlnet_hidden_states=controlnet_block_samples,
                        return_dict=False,
                    )[0]
                    
                    noise_uncond, noise_text = torch.chunk(model_out, 2, dim=0)
                    noise_guided = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                    x_t = self.noise_scheduler.step(noise_guided, timestep, x_t).prev_sample
            
            # デコード
            output_image = latent_to_pil(self.vae, x_t)[0]
            output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)
        
        return frames

if __name__ == "__main__":
    from PIL import Image

    model = SD3Model()
    input_image = Image.open("images_test/image_012.jpg").convert("RGB")
    train_prompt = "A car"
    inference_prompt = "A heavily rusted car"
    negative_prompt = ""
    guidance_scale = 4
    num_frames = 5

    frames = model(
        input_image=input_image,
        train_prompt=train_prompt,
        inference_prompt=inference_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
    )

    for idx, frame in enumerate(frames):
        frame.save(f"output_{idx}.png")
