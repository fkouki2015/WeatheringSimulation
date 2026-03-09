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
from diffusers import StableDiffusion3Pipeline, SD3ControlNetModel

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
    image = pipeline.image_processor.preprocess(pil).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = pipeline.vae.encode(image).latent_dist.mean

    latent = (latent - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latent.to(device=device, dtype=dtype)

def latent_to_pil(pipeline, latent, dtype):
    latent = latent.to(device=latent.device, dtype=dtype)
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    with torch.inference_mode():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")


def control_pil_to_latent(pipeline, pil: Image.Image, device, dtype
):
    x = transforms.ToTensor()(pil).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * 255.0 * 0.5 + 0.5
    with torch.no_grad():
        latents = pipeline.vae.encode(x).latent_dist.sample()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latents.to(device=device, dtype=dtype)

def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1



# ==========================================
# 経年変化モデル (SD3.5対応)
# ==========================================

class SD3Model(nn.Module):
    # デフォルト定数
    RESOLUTION = (1024, 1024)
    RANK = 256
    LEARNING_RATE = 1e-5
    TRAIN_STEPS = 500
    PRETRAINED_MODEL = "stabilityai/stable-diffusion-3.5-large"
    CONTROLNET_PATH = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
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
        self.dtype = torch.bfloat16

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.PRETRAINED_MODEL,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.noise_scheduler = self.pipeline.scheduler
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.text_encoder_3 = self.pipeline.text_encoder_3
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.tokenizer_3 = self.pipeline.tokenizer_3
        # ControlNet
        self.controlnet = SD3ControlNetModel.from_pretrained(
            self.CONTROLNET_PATH, torch_dtype=self.dtype
        ).to(self.device)

        # デフォルトで重みを固定
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)
        self.controlnet.requires_grad_(False)


    def _setup_training(self):
        """トレーニング用にLoRAとオプティマイザを設定"""
        transformer_lora_config = LoraConfig(
            r=self.RANK,
            lora_alpha=self.RANK,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.to_out.0", "attn.to_v", "attn.to_q", "attn.to_k"
            ]
        )
        self.transformer.add_adapter(transformer_lora_config)
        # self.controlnet.add_adapter(transformer_lora_config)
        self.transformer.train()
        # self.controlnet.train()
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

    def train_model(self, x_0, train_prompt):
        self._setup_training()

        prompt_embeds, neg_prompt_embeds, pooled_embeds, neg_pooled_embeds = self.pipeline.encode_prompt(
            prompt=[train_prompt],
            prompt_2=[train_prompt],
            prompt_3=[train_prompt],
            device=self.device,
            do_classifier_free_guidance=False,
        )

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for step in progress_bar:
            z = torch.randn_like(x_0)
            # t_scaler = sample_t_logit_normal(batch_size=x_0.shape[0], device=x_0.device)
            t_scaler = torch.rand(1).to(x_0.device) # 0.0-1.0の一様乱数
            t = t_scaler.view(-1, 1, 1, 1).to(self.dtype)
            x_t = (1.0 - t) * x_0 + t * z
            
            # ControlNet順伝播
            controlnet_block_samples = self.controlnet(
                hidden_states=self.transformer.pos_embed(x_t),
                controlnet_cond=self.control_latent,
                timestep=t_scaler * 1000,
                pooled_projections=pooled_embeds,
                conditioning_scale=1.0,
                return_dict=False,
            )[0]
            
            # Transformer順伝播
            pred_v = self.transformer(
                hidden_states=x_t,
                timestep=t_scaler * 1000,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                block_controlnet_hidden_states=controlnet_block_samples,
                return_dict=False,
            )[0]
            target = z - x_0
            loss = torch.nn.functional.mse_loss(pred_v.float(), target.float(), reduction="mean")
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
        
        control_image = canny_process(input_image, self.device, self.dtype)
        self.control_latent = control_pil_to_latent(self.pipeline, control_image, self.device, self.dtype)
        

        # 1. モデルのトレーニング
        self.train_model(x_0, train_prompt)
        
        # 2. 推論のセットアップ
        torch.manual_seed(1234)
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.text_encoder_3.eval()
        self.controlnet.eval()
        
        # テキスト埋め込み
        pos_prompt_embeds, neg_prompt_embeds, pos_pooled_embeds, neg_pooled_embeds = self.pipeline.encode_prompt(
            prompt=[inference_prompt],
            prompt_2=[inference_prompt],
            prompt_3=[inference_prompt],
            negative_prompt=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            device=self.device,
            do_classifier_free_guidance=True,
        )

        prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0)
        pooled_embeds = torch.cat([neg_pooled_embeds, pos_pooled_embeds], dim=0)

        frames = []
        # 3. 生成ループ (フレームごと)
        for i in range(num_frames):
            # sigmas = np.linspace(1.0, 1 / self.INFER_STEPS, self.INFER_STEPS)
            self.noise_scheduler.set_timesteps(self.INFER_STEPS, device=self.device)
            timesteps = self.noise_scheduler.timesteps # 0..1000のうち、INFER_STEPS個を等間隔で選択したもの
            
            # t_index計算
            normalized_i = (i + 1) / (num_frames)
            t_index = int((self.INFER_STEPS - 1) * (1.0 - normalized_i)**2.0)
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            # t_index: 0...INFER_STEPS-1

            t_scaler = timesteps[t_index] / self.noise_scheduler.num_train_timesteps
            t = t_scaler.view(-1, 1, 1, 1).to(self.dtype)
            z = torch.randn_like(x_0)
            x_t = (1.0 - t) * x_0 + t * z
            
            # デノイズ
            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i+1}/{num_frames}")
            for j, index in enumerate(progress):
                with torch.no_grad():
                    timestep = timesteps[index].to(self.device) # 0..1000
                    timestep_in = timestep.repeat(2)
                
                    x_t_in = x_t.repeat(2, 1, 1, 1)

                    control_in = self.control_latent.repeat(2, 1, 1, 1)

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
            output_image = latent_to_pil(self.pipeline, x_t, self.dtype)[0]
            output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)
        
        return frames

if __name__ == "__main__":
    from PIL import Image
    if not os.path.exists("temp_saves"):
        os.makedirs("temp_saves")

    model = SD3Model()
    input_image = Image.open("images_test/image_004.jpg").convert("RGB")
    train_prompt = "A camera"
    inference_prompt = "A heavily rusted camera"
    negative_prompt = ""
    guidance_scale = 4
    num_frames = 10

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
        frame.save(f"temp_saves/sd3_{date}_{idx}.png")
