import os
import sys
import datetime
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
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

# ==========================================
# ユーティリティ関数
# ==========================================
def canny_process(image, device, dtype):
    image_u8 = np.array(image)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    canny_img = cv2.Canny(gray, 100, 200)
    
    control_image = Image.fromarray(canny_img).convert("RGB")
    return control_image


def pil_to_latent(pipeline, pil: Image.Image, device, dtype):
    # latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(device=device, dtype=dtype)
    # latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(device=device, dtype=dtype)

    image = pipeline.image_processor.preprocess(pil).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        latent = pipeline.vae.encode(image).latent_dist.sample()
    latent = latent * pipeline.vae.config.scaling_factor # VAEのスケーリングを考慮
    return latent.to(device=device, dtype=dtype) # shape=(1, 4, H/8, W/8) dtype=pipeline.vae.dtype

def latent_to_pil(pipeline, latent):
    # latents_mean = (torch.tensor(pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latent.device, latent.dtype))
    # latents_std = (torch.tensor(pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latent.device, latent.dtype))
    
    latent = latent.to(device=latent.device, dtype=torch.float32)
    # latent = latents * latents_std / pipeline.vae.config.scaling_factor + latents_mean
    latent = latent / pipeline.vae.config.scaling_factor # VAEのスケーリングを考慮
    with torch.inference_mode():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")

def prepare_control_image(pipeline, pil: Image.Image, device, dtype):
    control_image = canny_process(pil, device, dtype)
    control_image = pipeline.control_image_processor.preprocess(control_image).to(device=device, dtype=dtype)
    return control_image


def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1



# ==========================================
# 経年変化モデル (SDXL対応)
# ==========================================

class SDXLModel(nn.Module):
    # デフォルト定数
    RESOLUTION = (1024, 1024)
    RANK = 256
    LEARNING_RATE = 1e-5
    TRAIN_STEPS = 200
    PRETRAINED_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_PATH = "diffusers/controlnet-canny-sdxl-1.0"
    DEVICE = "cuda"
    
    # 評価設定
    CLIP_EVAL_INTERVAL = 50
    CLIP_EVAL_STEPS = 20
    PERCEPTUAL_THRESHOLD = 0.05
    PERCEPTUAL_PATIENCE = 2
    
    # 推論設定
    INFER_STEPS = 50
    NOISE_RATIO = 0.6
    MAX_SEQUENCE_LENGTH = 256
    
    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.float32
        

        # ControlNet を先にロードしてからパイプラインに渡す
        self.controlnet = ControlNetModel.from_pretrained(
            self.CONTROLNET_PATH, torch_dtype=self.dtype
        ).to(self.device)

        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.PRETRAINED_MODEL,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.noise_scheduler = self.pipeline.scheduler
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2

        self.vae.to(self.device, dtype=torch.float32) # VAEはfloat32で動かす

        # デフォルトで重みを固定
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.controlnet.requires_grad_(False)


    def _setup_training(self):
        """トレーニング用にLoRAとオプティマイザを設定"""
        # unet_lora_config = LoraConfig(
        #     r=self.RANK,
        #     lora_alpha=self.RANK,
        #     init_lora_weights="gaussian",
        #     target_modules=[
        #         "to_out.0", "to_v", "to_q", "to_k"
        #     ]
        # )
        # self.unet.add_adapter(unet_lora_config)
        # self.controlnet.add_adapter(unet_lora_config)
        self.unet.train()
        # self.controlnet.train()
        self.unet.requires_grad_(True)
        # self.controlnet.requires_grad_(True)
        
        groups_by_scale = {}
        
        def _add_param(param, scale: float):
            key = round(float(scale), 6)
            if key not in groups_by_scale:
                groups_by_scale[key] = []
            groups_by_scale[key].append(param)
        
        for n, p in self.unet.named_parameters():
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

    def train_model(self, x_0, train_prompt, add_time_ids):
        self._setup_training()

        prompt_embeds, neg_prompt_embeds, pooled_embeds, neg_pooled_embeds = self.pipeline.encode_prompt(
            prompt=train_prompt,
            device=self.device,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        pooled_embeds = pooled_embeds.to(device=self.device, dtype=self.dtype)

        add_time_ids = add_time_ids.to(device=self.device, dtype=self.dtype)

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for step in progress_bar:
            z = torch.randn_like(x_0)
            timestep = torch.randint(0, 1000, (1,), device=self.device, dtype=self.dtype) # shape=(batch_size,)
            # t = t_scaler.view(-1, 1, 1, 1).to(self.dtype)
            # x_t = (1.0 - t) * x_0 + t * z
            x_t = self.noise_scheduler.add_noise(x_0, z, timestep)
            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
            
            # down_block_samples, mid_block_sample = self.controlnet(
            #     x_t,
            #     timestep,
            #     encoder_hidden_states=prompt_embeds,
            #     controlnet_cond=self.control_image,
            #     conditioning_scale=0.8,
            #     added_cond_kwargs=added_cond_kwargs,
            #     return_dict=False,
            # )
            
            model_pred = self.unet(
                x_t,
                timestep,
                encoder_hidden_states=prompt_embeds,
                # down_block_additional_residuals=down_block_samples,
                # mid_block_additional_residual=mid_block_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            loss = torch.nn.functional.mse_loss(model_pred.float(), z.float(), reduction="mean")
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
        input_image = input_image.resize(self.RESOLUTION, Image.LANCZOS)
        x_0 = pil_to_latent(self.pipeline, input_image, self.device, self.dtype)
        
        self.control_image = prepare_control_image(self.pipeline, input_image, self.device, self.dtype)

        add_time_ids =  self.pipeline._get_add_time_ids(
            original_size=self.RESOLUTION,
            crops_coords_top_left=(0,0),
            target_size=self.RESOLUTION,
            dtype=self.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        )

        add_neg_time_ids =  self.pipeline._get_add_time_ids(
            original_size=self.RESOLUTION,
            crops_coords_top_left=(0,0),
            target_size=self.RESOLUTION,
            dtype=self.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        )
        

        # 1. モデルのトレーニング
        self.train_model(x_0, train_prompt, add_time_ids)
        
        # 2. 推論のセットアップ
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.controlnet.eval()
        
        # テキスト埋め込み
        pos_prompt_embeds, neg_prompt_embeds, pos_pooled_embeds, neg_pooled_embeds = self.pipeline.encode_prompt(
            prompt=inference_prompt,
            negative_prompt=negative_prompt,
            device=self.device,
            do_classifier_free_guidance=True,
        )
        prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0).to(device=self.device, dtype=self.dtype)
        pooled_embeds = torch.cat([neg_pooled_embeds, pos_pooled_embeds], dim=0).to(device=self.device, dtype=self.dtype)

        
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0).to(device=self.device, dtype=self.dtype)

        frames = []
        # 3. 生成ループ (フレームごと)
        for i in range(num_frames):
            torch.manual_seed(1234)
            self.noise_scheduler.set_timesteps(self.INFER_STEPS, device=self.device)
            timesteps = self.noise_scheduler.timesteps # 0..1000のうち、INFER_STEPS個を等間隔で選択したもの
            
            # t_index計算
            normalized_i = (i + 1) / (num_frames)
            t_index = int((self.INFER_STEPS - 1) * (1.0 - normalized_i)**2.0)
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            # t_index: 0...INFER_STEPS-1
            t_index = 0

            z = torch.randn_like(x_0)
            x_t = self.noise_scheduler.add_noise(x_0, z, timesteps[t_index].unsqueeze(0))
            # x_t = self.pipeline.prepare_latents(1, 4, self.RESOLUTION[0], self.RESOLUTION[1], self.dtype, device=self.device, generator=torch.Generator(device=self.device).manual_seed(1234))
            
            # デノイズ
            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i+1}/{num_frames}")
            for j, index in enumerate(progress):
                with torch.no_grad():
                    timestep = timesteps[index].to(device=self.device) # 0..1000
                    added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
                    
                    x_t_in = x_t.repeat(2, 1, 1, 1)
                    x_t_in = self.noise_scheduler.scale_model_input(x_t_in, timestep)
                    control_in = self.control_image.repeat(2, 1, 1, 1)

                    down_block_samples, mid_block_sample = self.controlnet(
                        x_t_in,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=control_in,
                        conditioning_scale=0.8,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )
                    
                    model_pred = self.unet(
                        x_t_in,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        # down_block_additional_residuals=down_block_samples,
                        # mid_block_additional_residual=mid_block_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    
                    noise_uncond, noise_text = torch.chunk(model_pred, 2, dim=0)
                    noise_guided = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                    x_t = self.noise_scheduler.step(noise_guided, timestep, x_t, return_dict=False)[0]
            
            # デコード
            output_image = latent_to_pil(self.pipeline, x_t)[0]
            output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)
        
        return frames

if __name__ == "__main__":
    from PIL import Image
    if not os.path.exists("temp_saves"):
        os.makedirs("temp_saves")

    model = SDXLModel()
    input_image = Image.open("images_test/image_012.jpg").convert("RGB")
    train_prompt = "A car"
    inference_prompt = "A fully rusted car"
    negative_prompt = "clean, new"
    guidance_scale = 7
    num_frames = 1

    frames = model(
        input_image=input_image,
        train_prompt=train_prompt,
        inference_prompt=inference_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
    )

    date = datetime.datetime.now().strftime("%m%d_%H%M")
    for idx, frame in enumerate(frames):
        frame.save(f"temp_saves/sdxl_{date}_{idx}.png")
