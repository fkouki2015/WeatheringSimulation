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
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
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
    """PIL画像を潜在空間にエンコード (SD3用)"""
    to_tensor = transforms.ToTensor()
    x = to_tensor(pil).unsqueeze(0).to(device)
    x = (x * 2 - 1).to(dtype=torch.float32)
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        latents = posterior.sample()

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


def get_sigmas(scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    """シグマ値を取得"""
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(device)]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def import_model_class(pretrained_model_name_or_path: str, subfolder="text_encoder", local_files_only=False):
    """モデル設定に基づいて適切なテキストエンコーダクラスをインポート"""
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, local_files_only=local_files_only
    )
    model_class_name = config.architectures[0]
    if model_class_name == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class_name == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported model class: {model_class_name}")


def tokenize_prompt(tokenizer, prompt_list, max_length, device):
    """プロンプトをトークナイズ"""
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids.to(device)


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt_list,
    device,
    weight_dtype,
    text_input_ids=None,
    num_images_per_prompt=1,
):
    """CLIPテキストエンコーダでプロンプトをエンコード"""
    batch_size = len(prompt_list)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
    elif text_input_ids is None:
        raise ValueError("Either tokenizer or text_input_ids must be provided")
    else:
        text_input_ids = text_input_ids.to(device)

    outputs = text_encoder(text_input_ids, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-2]
    pooled = getattr(outputs, "text_embeds", None)
    if pooled is None:
        pooled = outputs.pooler_output
    prompt_embeds = last_hidden.to(dtype=weight_dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled.to(device=device, dtype=weight_dtype)


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt_list,
    max_sequence_length,
    num_images_per_prompt,
    device,
    weight_dtype,
    text_input_ids=None,
):
    """T5テキストエンコーダでプロンプトをエンコード"""
    batch_size = len(prompt_list)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
    elif text_input_ids is None:
        raise ValueError("Either tokenizer or text_input_ids must be provided")
    else:
        text_input_ids = text_input_ids.to(device)

    prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt_list,
    max_sequence_length,
    device,
    weight_dtype,
    num_images_per_prompt=1,
    text_input_ids_list=None,
):
    """3つのテキストエンコーダ（CLIP x2 + T5）でプロンプトをエンコード"""
    clip_tokenizers, clip_encoders = tokenizers[:2], text_encoders[:2]
    clip_embeds_list, pooled_list = [], []

    for i, (tok, enc) in enumerate(zip(clip_tokenizers, clip_encoders)):
        if tok is not None:
            token_ids = tokenize_prompt(tok, prompt_list, 77, device)
        else:
            token_ids = (
                text_input_ids_list[i].to(device)
                if text_input_ids_list and text_input_ids_list[i] is not None
                else None
            )
            if token_ids is None:
                raise ValueError(f"No tokenizer or token IDs provided for CLIP encoder {i+1}")

        embeds, pooled = _encode_prompt_with_clip(
            text_encoder=enc,
            tokenizer=None,
            prompt_list=prompt_list,
            device=device,
            weight_dtype=weight_dtype,
            text_input_ids=token_ids,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_embeds_list.append(embeds)
        pooled_list.append(pooled)

    clip_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_embeds = torch.cat(pooled_list, dim=-1)

    if tokenizers[2] is not None:
        t5_token_ids = tokenize_prompt(
            tokenizers[2], prompt_list, max_sequence_length, device
        )
    else:
        t5_token_ids = (
            text_input_ids_list[2].to(device)
            if text_input_ids_list and text_input_ids_list[2] is not None
            else None
        )
        if t5_token_ids is None:
            raise ValueError("No tokenizer or token IDs provided for T5 encoder")

    t5_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[2],
        tokenizer=None,
        prompt_list=prompt_list,
        max_sequence_length=max_sequence_length,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        weight_dtype=weight_dtype,
        text_input_ids=t5_token_ids,
    )

    t5_dim = t5_embeds.shape[-1]
    clip_embeds = torch.nn.functional.pad(
        clip_embeds,
        (0, t5_dim - clip_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

    return prompt_embeds, pooled_embeds


# ==========================================
# 経年変化モデル (SD3.5対応)
# ==========================================

class SD3Model(nn.Module):
    # デフォルト定数
    RESOLUTION = (1024, 1024)
    RANK = 8
    LEARNING_RATE = 1e-5
    TRAIN_STEPS = 600
    PRETRAINED_MODEL = "stabilityai/stable-diffusion-3.5-medium"
    CONTROLNET_PATH = "InstantX/SD3-Controlnet-Canny"
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
        self._init_models()
        
    def _init_models(self):
        """すべての拡散モデルとコンポーネントを初期化"""
        # スケジューラ
        self.noise_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="scheduler", local_files_only=True
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        
        # トークナイザ
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer", local_files_only=True
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer_2", local_files_only=True
        )
        self.tokenizer_three = T5TokenizerFast.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer_3", local_files_only=True
        )
        
        # テキストエンコーダ
        text_encoder_cls_one = import_model_class(self.PRETRAINED_MODEL, subfolder="text_encoder")
        text_encoder_cls_two = import_model_class(self.PRETRAINED_MODEL, subfolder="text_encoder_2")
        text_encoder_cls_three = import_model_class(self.PRETRAINED_MODEL, subfolder="text_encoder_3")
        
        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder", local_files_only=True
        )
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder_2", local_files_only=True
        )
        self.text_encoder_three = text_encoder_cls_three.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder_3", local_files_only=True
        )
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="vae", local_files_only=True
        )
        
        # Transformer (SD3用)
        self.transformer = SD3Transformer2DModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="transformer", local_files_only=True
        )
        
        # ControlNet
        self.controlnet = SD3ControlNetModel.from_pretrained(
            self.CONTROLNET_PATH, torch_dtype=self.dtype, local_files_only=True
        )
        self.controlnet.to(device=self.device, dtype=self.dtype)
        self.controlnet.requires_grad_(False)
        
        # LPIPS
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()

        # デバイスへ移動
        self.vae.to(device=self.device, dtype=torch.float32)
        self.transformer.to(device=self.device, dtype=self.dtype)
        self.text_encoder_one.to(device=self.device, dtype=self.dtype)
        self.text_encoder_two.to(device=self.device, dtype=self.dtype)
        self.text_encoder_three.to(device=self.device, dtype=self.dtype)

        # デフォルトで重みを固定
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.text_encoder_three.requires_grad_(False)

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
        
        groups_by_scale = {}
        
        def _add_param(param, scale: float):
            key = round(float(scale), 6)
            if key not in groups_by_scale:
                groups_by_scale[key] = []
            groups_by_scale[key].append(param)
        
        for n, p in self.transformer.named_parameters():
            if p.requires_grad:
                _add_param(p, 1.0)
        
        param_groups = [
            {"params": ps, "lr": self.LEARNING_RATE * scale}
            for scale, ps in groups_by_scale.items()
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )
        
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=10,
            num_training_steps=self.TRAIN_STEPS,
        )
        
        return [p for g in param_groups for p in g["params"]]

    def _apply_controlnet(self, hidden_states, timestep, encoder_hidden_states,
                          pooled_projections, controlnet_cond, conditioning_scale=1.0):
        """SD3 ControlNetを適用してblock samplesを返す"""
        controlnet_block_samples = self.controlnet(
            hidden_states=hidden_states,
            controlnet_cond=controlnet_cond,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )[0]
        return controlnet_block_samples

    def train_model(self):
        """LoRAを使用してモデルをファインチューニング"""
        image = self.input_image
        
        trainable_params = self._setup_training()
        
        latent = pil_to_latent(self.vae, image, self.device, self.dtype)
        
        self.transformer.train()
        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        
        for step in progress_bar:
            # テキスト埋め込みの計算
            cond_embeds, cond_pooled = encode_prompt(
                [self.text_encoder_one, self.text_encoder_two, self.text_encoder_three],
                [self.tokenizer_one, self.tokenizer_two, self.tokenizer_three],
                [self.train_prompt],
                max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                device=self.device,
                weight_dtype=self.dtype,
            )
            uncond_embeds, uncond_pooled = encode_prompt(
                [self.text_encoder_one, self.text_encoder_two, self.text_encoder_three],
                [self.tokenizer_one, self.tokenizer_two, self.tokenizer_three],
                [""],
                max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                device=self.device,
                weight_dtype=self.dtype,
            )
            
            # CFG dropout
            if random.random() < 0.15:
                prompt_embeds = uncond_embeds
                pooled_embeds = uncond_pooled
            else:
                prompt_embeds = cond_embeds
                pooled_embeds = cond_pooled
            
            # ノイズ追加
            noise = torch.randn_like(latent)
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=1,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latent.device)
            
            sigmas = get_sigmas(self.noise_scheduler_copy, timesteps, n_dim=latent.ndim, dtype=latent.dtype, device=latent.device)
            noisy_latent = (1.0 - sigmas) * latent + sigmas * noise
            
            # ControlNet順伝播
            controlnet_block_samples = self._apply_controlnet(
                hidden_states=noisy_latent,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                controlnet_cond=self.control_image_tensor,
            )
            
            # Transformer順伝播
            model_pred_raw = self.transformer(
                hidden_states=noisy_latent,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                block_controlnet_hidden_states=controlnet_block_samples,
                return_dict=False,
            )[0]
            model_pred = model_pred_raw * (-sigmas) + noisy_latent
            target = latent
            
            # 損失計算
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme="logit_normal", sigmas=sigmas
            )
            
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            
            # 最適化
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})

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
        
        # 制御画像を前処理（VAEで潜在空間にエンコード: SD3 ControlNetは16chの潜在入力を期待）
        self.control_image = canny_process(self.input_image, self.device, self.dtype)
        self.control_image_tensor = pil_to_latent(self.vae, self.control_image, self.device, self.dtype)

        # 1. モデルのトレーニング
        self.train_model()
        
        # 2. 推論のセットアップ
        torch.manual_seed(1234)
        self.vae.eval()
        self.transformer.eval()
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
            weight_dtype=self.dtype,
        )
        
        # 潜在変数の準備
        base_latent = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)
        control_tensor = self.control_image_tensor
        
        frames = []
        
        # 3. 生成ループ (フレームごと)
        for i in range(num_frames):
            torch.manual_seed(1234)
            scheduler = copy.deepcopy(self.noise_scheduler)
            scheduler.set_timesteps(self.INFER_STEPS, device=self.device)
            timesteps = scheduler.timesteps
            
            # t_index計算
            normalized_i = i / max(1, num_frames - 1)
            t_index = self.INFER_STEPS - int((self.INFER_STEPS - 1) * (normalized_i ** self.NOISE_RATIO)) - 1
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            if num_frames == 1:
                t_index = 0
            
            # ノイズ追加
            sigma = get_sigmas(scheduler, timesteps[t_index], n_dim=base_latent.ndim, dtype=base_latent.dtype, device=base_latent.device)
            noise = torch.randn_like(base_latent)
            latents = (1.0 - sigma) * base_latent + sigma * noise
            
            # デノイズ
            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i+1}/{num_frames}")
            for j, index in enumerate(progress):
                t = timesteps[index].to(self.device)
                t_in = t.repeat(2)
                
                with torch.no_grad():
                    latents_in = latents.repeat(2, 1, 1, 1)
                    control_in = control_tensor.repeat(2, 1, 1, 1)
                    
                    controlnet_block_samples = self._apply_controlnet(
                        hidden_states=latents_in,
                        timestep=t_in,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        controlnet_cond=control_in,
                    )
                    
                    model_out = self.transformer(
                        hidden_states=latents_in,
                        timestep=t_in,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        block_controlnet_hidden_states=controlnet_block_samples,
                        return_dict=False,
                    )[0]
                    
                    noise_uncond, noise_text = torch.chunk(model_out, 2, dim=0)
                    noise_guided = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                    latents = scheduler.step(noise_guided, t, latents).prev_sample
            
            # デコード
            output_image = latent_to_pil(self.vae, latents)[0]
            output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)
        
        return frames
