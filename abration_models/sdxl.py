import os
import sys
import uuid
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision import transforms
from peft import LoraConfig
import lpips
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from diffusers.optimization import get_scheduler

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
    canny_img = cv2.Canny(gray, 80, 160)
    
    control_image = Image.fromarray(canny_img).convert("RGB")
    if isinstance(control_image, Image.Image):
        control_image = transforms.ToTensor()(control_image).unsqueeze(0)  # (1, 3, H, W)
    control_image = control_image.to(device=device, dtype=dtype)
    return control_image


def depth_process(image, device, dtype):
    """画像から深度マップを生成"""
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        
    d = depth_resized[0]
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    depth_u8 = (d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    depth_pil = Image.fromarray(depth_u8, mode="L")
    control_image = depth_pil.convert("RGB")
    control_tensor = transforms.ToTensor()(control_image).unsqueeze(0).to(device=device, dtype=dtype)
    return control_tensor


def pil_to_latent(vae, pil, device, dtype):
    """PIL画像を潜在空間にエンコード"""
    img = transforms.ToTensor()(pil).unsqueeze(0).to(device=device, dtype=dtype)
    img = img * 2 - 1
    with torch.no_grad():
        posterior = vae.encode(img).latent_dist
        z = posterior.mean * vae.config.scaling_factor
    return z


def latent_to_pil(vae, latents):
    """潜在ベクトルをPIL画像にデコード"""
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs = (imgs * 255).round().to(torch.uint8).cpu().permute(0,2,3,1).numpy()
    pil_list = [Image.fromarray(arr) for arr in imgs]
    return pil_list


def find_token_indices(tokenizer, input_ids_tensor, target_word: str):
    """ターゲット単語に対応するトークンのインデックスを検索"""
    ids = input_ids_tensor[0].tolist()
    out = []
    for i, tok_id in enumerate(ids):
        token_str = tokenizer.decode([tok_id]).strip().lower()
        if target_word in token_str:
            # print("ターゲットトークン検出:", i, tok_id, token_str)
            out.append(i)
    if len(out) == 0:
        print("ターゲットトークン未検出:", target_word)
    return out


# ==========================================
# 経年変化モデル
# ==========================================

class WeatheringModel(nn.Module):
    # デフォルト定数
    RESOLUTION = (1024, 1024)  # SDXLの標準解像度
    RANK = 8
    LEARNING_RATE = 1e-5
    TRAIN_STEPS = 600
    PRETRAINED_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL
    CONTROLNET_PATH_CANNY = "diffusers/controlnet-canny-sdxl-1.0"  # SDXL用ControlNet
    CONTROLNET_STRENGTHS = [1.0]
    DEVICE = "cuda"
    LORA_DROPOUT = 0.0
    
    # 評価設定
    CLIP_EVAL_INTERVAL = 50
    CLIP_EVAL_STEPS = 20
    PERCEPTUAL_THRESHOLD = 0.05
    PERCEPTUAL_PATIENCE = 2
    
    # 推論設定
    INFER_STEPS = 50
    NOISE_RATIO = 1.0
    
    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.float32
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self._init_models()
        
    def _init_models(self):
        """すべての拡散モデルとコンポーネントを初期化"""
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="scheduler"
        )
        self.ddpm_scheduler = DDPMScheduler.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="scheduler"
        )
        
        # SDXLは2つのテキストエンコーダを使用
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer_2"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder"
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder_2"
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="unet"
        )
        self.controlnet_canny = ControlNetModel.from_pretrained(
            self.CONTROLNET_PATH_CANNY, torch_dtype=torch.float32
        )
        self.controlnets = [self.controlnet_canny] # ControlNetを追加する場合ここに追加
        
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()

        # デバイスへ移動
        self.vae.to(device=self.device, dtype=self.dtype)
        self.text_encoder.to(device=self.device)
        self.text_encoder_2.to(device=self.device)
        self.unet.to(device=self.device)
        for c in self.controlnets:
            c.to(device=self.device)

        # デフォルトで重みを固定
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        for c in self.controlnets:
            c.requires_grad_(False)

    def _encode_prompt_sdxl(self, prompt):
        """SDXLの2つのテキストエンコーダを使用してプロンプトをエンコード"""
        # 最初のテキストエンコーダ
        tokens = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = self.tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            encoder_output = self.text_encoder(tokens.input_ids.to(self.device), output_hidden_states=True)
            text_embeds = encoder_output.hidden_states[-2]
            
            encoder_output_2 = self.text_encoder_2(tokens_2.input_ids.to(self.device), output_hidden_states=True)
            pooled_text_embeds = encoder_output_2[0]
            text_embeds_2 = encoder_output_2.hidden_states[-2]
        
        # 2つのエンコーダからの埋め込みを連結
        text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)
        return text_embeds, pooled_text_embeds


    def _setup_training(self):
        """トレーニング用にLoRAとオプティマイザを設定"""
        self.unet.requires_grad_(False)
        self.unet.train()

        for c in self.controlnets:
            c.requires_grad_(True)
            c.train()
        
        unet_lora_config = LoraConfig(
            r=self.RANK,
            lora_alpha=self.RANK,
            init_lora_weights="gaussian",
            target_modules=["attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"],
            lora_dropout=self.LORA_DROPOUT
        )
        self.adapter_name = f"train-{uuid.uuid4().hex[:8]}"
        self.unet.add_adapter(unet_lora_config, adapter_name=self.adapter_name)
        self.unet.set_adapters([self.adapter_name])
        
        base_lr = self.LEARNING_RATE
        groups_by_scale = {}
        
        # LRスケールごとにパラメータをグループ化
        def _add_param(param, scale: float):
            key = round(float(scale), 6)
            if key not in groups_by_scale:
                groups_by_scale[key] = []
            groups_by_scale[key].append(param)
        
        # UNetとControlNetに異なる学習率を設定
        for n, p in self.unet.named_parameters():
            if p.requires_grad:
                if "attn2" in n:
                    _add_param(p, 1.0)
        
        for c in self.controlnets:
            for n, p in c.named_parameters():
                if p.requires_grad:
                    _add_param(p, 0.07)
        
        param_groups = [
            {"params": ps, "lr": base_lr * scale}
            for scale, ps in groups_by_scale.items()
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )
        
        return [p for g in param_groups for p in g["params"]]
        
    def _apply_controlnets(self, latents_in, t, text_embeddings, control_images, strengths=None):
        """複数のControlNetを適用可能"""
        combined_down = None
        combined_mid = None
        
        for idx, cn in enumerate(self.controlnets):
            down_res, mid_res = cn(
                latents_in,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_images[idx],
                return_dict=False,
            )
            s = float(strengths[idx] if strengths else 1.0)

            if combined_down is None:
                combined_down = [(s * res).to(dtype=self.dtype) for res in down_res]
                combined_mid = (mid_res * s).to(dtype=self.dtype)
            else:
                combined_down = [
                    prev + (curr * s).to(dtype=self.dtype) 
                    for prev, curr in zip(combined_down, down_res)
                ]
                combined_mid = combined_mid + (mid_res * s).to(dtype=self.dtype)
                
        return combined_down, combined_mid

    def _generate_preview_image(self, latent, prompt, seed, control_images):
        """単一のプレビュー画像を生成"""
        self.unet.eval()
        for c in self.controlnets:
            c.eval()
            
        height, width = self.RESOLUTION[1], self.RESOLUTION[0]
        h, w = height // 8, width // 8

        # テキスト埋め込みの準備 (SDXL dual encoders)
        cond, _ = self._encode_prompt_sdxl(prompt)
        uncond, _ = self._encode_prompt_sdxl("")
        text_embeds = torch.cat([uncond, cond], dim=0)

        # スケジューラの準備
        scheduler = DDIMScheduler.from_pretrained(self.PRETRAINED_MODEL, subfolder="scheduler")
        scheduler.set_timesteps(self.CLIP_EVAL_STEPS, device=self.device)
        
        generator = torch.Generator(device=self.device.type).manual_seed(seed)
        noise = torch.randn(1, 4, h, w, device=self.device, dtype=self.dtype, generator=generator)
        latents = scheduler.add_noise(latent, noise, scheduler.timesteps[0])

        # デノイズループ
        for t in scheduler.timesteps:
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda"), dtype=self.dtype):
                lat_in = torch.cat([latents] * 2, dim=0)
                lat_in = scheduler.scale_model_input(lat_in, t)

                # 制御画像の準備
                control_imgs_in = []
                for ci in control_images:
                    control_imgs_in.append(ci.repeat(2, 1, 1, 1) if ci.shape[0] == 1 else ci)
                
                down_res, mid_res = self._apply_controlnets(
                    lat_in, t, text_embeds, control_imgs_in, strengths=self.CONTROLNET_STRENGTHS
                )
                
                noise_pred = self.unet(
                    lat_in, t, text_embeds,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample

                eps_u, eps_c = noise_pred.chunk(2, dim=0)
                # プレビュー用の固定ガイダンススケール
                eps = eps_u + 1.0 * (eps_c - eps_u)
                latents = scheduler.step(eps, t, latents).prev_sample

        pil = latent_to_pil(self.vae, latents)[0]
        self.unet.train() # トレーニングモードに戻す
        return pil

    def _evaluate(self, step, latent, control_images, image):
        """LPIPSを使用して現在のモデルパフォーマンスを評価する。"""
        with torch.no_grad():
            dists = []
            # シードを変えて3回プレビューを生成
            for k in range(3):
                preview_pil = self._generate_preview_image(
                    latent=latent,
                    prompt=self.train_prompt,
                    seed=step * 100 + k,
                    control_images=control_images
                )
                # 知覚的距離を計算
                dist = compute_perceptual_distance(
                    preview_pil, image, self.lpips_model, self.device
                )
                dists.append(dist)
            
            return sum(dists) / len(dists)

    def train_model(self):
        """LoRAを使用してモデルをファインチューニング"""
        image = self.input_image
        
        trainable_params = self._setup_training()
        
        # SDXL dual text encoders
        cond_emb, _ = self._encode_prompt_sdxl(self.train_prompt)
        
        timesteps = self.ddpm_scheduler.timesteps
        prev_lpips = None
        no_improve_streak = 0
        
        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        
        for step in progress_bar:
            # 1. 入力の準備
            latent = pil_to_latent(self.vae, image, self.device, self.dtype)
            bsz = latent.shape[0]
            t = torch.randint(0, len(timesteps)-1, (bsz,), device=self.device, dtype=torch.long)
            
            noise = torch.randn_like(latent, device=self.device, dtype=self.dtype)
            noisy_latent = self.ddpm_scheduler.add_noise(latent, noise, t)
            
            # 2. ControlNet 順伝播
            down_res, mid_res = self._apply_controlnets(
                noisy_latent, t, cond_emb, self.control_images, strengths=[1.0] * len(self.control_images)
            )
            
            # 3. UNet 順伝播
            model_pred = self.unet(
                noisy_latent, t, cond_emb,
                down_block_additional_residuals=[s.to(dtype=self.dtype) for s in down_res],
                mid_block_additional_residual=mid_res.to(dtype=self.dtype),
            ).sample
            
            # 4. 最適化
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # 5. 評価と早期停止
            if step % self.CLIP_EVAL_INTERVAL == 0:
                avg_dist = self._evaluate(step, latent, self.control_images, image)
                
                if prev_lpips is None:
                    improvement_rate = 0.0
                    no_improve_streak = 0
                else:
                    improvement = prev_lpips - avg_dist
                    improvement_rate = improvement / (prev_lpips + 1e-12)
                    
                    if improvement_rate >= self.PERCEPTUAL_THRESHOLD:
                        no_improve_streak = 0
                    else:
                        no_improve_streak += 1
                        if no_improve_streak >= self.PERCEPTUAL_PATIENCE:
                            tqdm.write(f"早期停止: 改善率 {improvement_rate:.4f} < {self.PERCEPTUAL_THRESHOLD} が {self.PERCEPTUAL_PATIENCE} 回連続")
                            break
                
                prev_lpips = avg_dist
                tqdm.write(f"\nLPIPS={avg_dist:.4f}, 改善率={improvement_rate*100:.2f}%")
                sys.stdout.flush()

                
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

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
        self.input_image = input_image
        self.train_prompt = train_prompt
        # 制御画像を前処理
        self.control_images = []
        canny_image = canny_process(input_image, self.device, self.dtype)
        self.control_images.append(canny_image)
        # ControlNetの条件画像を追加する場合ここに追加
        # self.control_image_raw = transforms.ToTensor()(input_image).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        
        # 1. モデルのトレーニング
        self.train_model()
        
        # 2. 推論のセットアップ
        torch.manual_seed(1234)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        for c in self.controlnets:
            c.eval()
            
        # SDXL dual text encoders
        cond_emb, _ = self._encode_prompt_sdxl(inference_prompt)
        uncond_emb, _ = self._encode_prompt_sdxl(negative_prompt)
        text_embeddings = torch.cat([uncond_emb, cond_emb], dim=0)
        

        
        # 潜在変数の準備
        self.ddim_scheduler.set_timesteps(self.INFER_STEPS, device=self.device)
        timesteps = self.ddim_scheduler.timesteps
        start_latent = pil_to_latent(self.vae, input_image, self.device, self.dtype)
        noise = torch.randn_like(start_latent, device=self.device, dtype=self.dtype)
        
        frames = []
        
        # 3. 生成ループ (フレームごと)
        for i in range(num_frames):
            # 経年変化係数を計算 (フレームインデックスに基づいて 0.0 から 1.0)
            # スムーズな遷移効果を作成するため正弦波補間を使用
            normalized_i = (i + 1) / num_frames * math.pi / 2
            t_index = self.INFER_STEPS - int((self.INFER_STEPS - 1) * math.sin(normalized_i)) - 1
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            
            # 特定のタイムステップで元の画像と混合されたノイズから開始
            noisy_latent = self.ddim_scheduler.add_noise(start_latent, noise, timesteps[t_index])
            
            # デノイズ
            with torch.no_grad():
                latents = noisy_latent
                # タイムステップを反復
                for t in tqdm(timesteps[t_index:], desc=f"{i+1}/{num_frames}"):
                    with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda"), dtype=self.dtype):
                        
                        lat_in = torch.cat([latents] * 2, dim=0)
                        lat_in = self.ddim_scheduler.scale_model_input(lat_in, t)
                        
                        # 制御入力の準備
                        control_imgs_in = []
                        for ci in self.control_images:
                            control_imgs_in.append(ci.repeat(2, 1, 1, 1) if ci.shape[0] == 1 else ci)
                            
                        down_res, mid_res = self._apply_controlnets(
                            lat_in, t, text_embeddings, control_imgs_in, strengths=self.CONTROLNET_STRENGTHS
                        )
                        
                        noise_pred = self.unet(
                            lat_in, t, text_embeddings,
                            down_block_additional_residuals=down_res,
                            mid_block_additional_residual=mid_res,
                        ).sample
                        
                        # ガイダンスを実行
                        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                        
                        latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample
                
                output_image = latent_to_pil(self.vae, latents)[0]
                frames.append(output_image)
            
        return frames
