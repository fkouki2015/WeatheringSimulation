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
from transformers import CLIPTextModel, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.models.attention_processor import AttnProcessor 

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
# 経年変化ワード強調アテンション
# ==========================================

class AgingAttentionProcessor(AttnProcessor):
    def __init__(self, aging_token_indices, scale_max=2.0, zero_to_one=False):
        super().__init__()
        self.aging_token_indices = aging_token_indices
        self.scale_max = scale_max
        self.zero_to_one = zero_to_one
        self.current_factor = 0.0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        # Q, K, V 射影
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        bsz, q_len, _ = query.shape
        k_len = key.shape[1]
        head_dim = query.shape[-1] // attn.heads
        
        # マルチヘッドアテンション用にリシェイプ
        query = query.view(bsz, q_len, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(bsz, k_len, attn.heads, head_dim).transpose(1, 2)
        value = value.view(bsz, k_len, attn.heads, head_dim).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        attn_probs = attn_scores.softmax(dim=-1)

        # 経年変化エフェクト: 特定のトークンアテンションにスケーリングを適用
        if encoder_hidden_states is not hidden_states and bsz >= 2 and self.aging_token_indices and self.current_factor >= 0:
            cond_idx = 1  # [uncond, cond]
            if self.zero_to_one:
                # 0.0 から 1.0 の間で補間
                scale_mul = max(self.current_factor, 1e-6)
            else:
                # 1.0 から scale_max の間で補間
                scale_mul = 1.0 + self.current_factor * (self.scale_max - 1.0)
            
            attn_probs_cond = attn_probs[cond_idx]
            attn_probs_cond[:, :, self.aging_token_indices] *= scale_mul
            
            # 再正規化
            attn_probs_cond = attn_probs_cond / (attn_probs_cond.sum(dim=-1, keepdim=True) + 1e-8)
            attn_probs[cond_idx] = attn_probs_cond
            
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(bsz, q_len, attn.heads * head_dim)
        
        hidden_states = attn.to_out[0](hidden_states)
        if attn.to_out[1] is not None:
            hidden_states = attn.to_out[1](hidden_states)
            
        return hidden_states


# ==========================================
# 経年変化モデル
# ==========================================

class WeatheringModel(nn.Module):
    # デフォルト定数
    RESOLUTION = (512, 512)
    RANK = 8
    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_PATH_CANNY = "lllyasviel/sd-controlnet-canny"
    CONTROLNET_STRENGTHS = [1.0]
    DEVICE = "cuda"
    LORA_DROPOUT = 0.0
    
    # 二段階学習設定
    T_MID = 500              # Phase分割の境界タイムステップ
    PHASE1_STEPS = 100       # Phase 1: LoRA学習ステップ数
    PHASE2_STEPS = 300       # Phase 2: ControlNet学習ステップ数
    PHASE1_LR = 5e-5         # Phase 1 学習率
    PHASE2_LR = 1e-6         # Phase 2 学習率
    
    # 評価設定
    CLIP_EVAL_INTERVAL = 50
    CLIP_EVAL_STEPS = 20
    PERCEPTUAL_THRESHOLD = 0.05
    PERCEPTUAL_PATIENCE = 2
    
    # 推論設定
    INFER_STEPS = 50
    NOISE_RATIO = 1.0
    ATTN_SCALE_MAX = 10.0 # アテンション強調最大値
    ATTN_ZERO_TO_ONE = False # アテンションを0.0から1.0の間で補間
    LORA_SCALE_MIN = 0.3 # 最終フレームでのLoRAスケール (1.0→この値まで低減)
    
    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.float32
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self._init_models()
        
    def _init_models(self):
        """すべての拡散モデルとコンポーネントを初期化"""
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="scheduler", local_files_only=True
        )
        self.ddpm_scheduler = DDPMScheduler.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="scheduler", local_files_only=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="text_encoder", local_files_only=True
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="vae", local_files_only=True
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.PRETRAINED_MODEL, subfolder="unet", local_files_only=True
        )
        self.controlnet_canny = ControlNetModel.from_pretrained(
            self.CONTROLNET_PATH_CANNY, torch_dtype=torch.float32, local_files_only=True
        )
        self.controlnets = [self.controlnet_canny] # ControlNetを追加する場合ここに追加
        
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()

        # デバイスへ移動
        self.vae.to(device=self.device, dtype=self.dtype)
        self.text_encoder.to(device=self.device)
        self.unet.to(device=self.device)
        for c in self.controlnets:
            c.to(device=self.device)

        # デフォルトで重みを固定
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        for c in self.controlnets:
            c.requires_grad_(False)

    # ControlNetで学習する浅い層の割合 (0.0~1.0, 例: 0.5 = 前半のみ学習)
    CONTROLNET_TRAINABLE_RATIO = 0.5

    def _setup_phase1_lora(self):
        """Phase 1: LoRAのみ学習。ControlNetは完全に凍結。"""
        print("\n" + "="*50)
        print("Phase 1: LoRA学習セットアップ (高ノイズ領域)")
        print("="*50)
        
        self.unet.requires_grad_(False)
        self.unet.train()
        
        # ControlNetは完全凍結
        for c in self.controlnets:
            c.requires_grad_(False)
            c.eval()
        
        # LoRAアダプタを追加
        unet_lora_config = LoraConfig(
            r=self.RANK,
            lora_alpha=self.RANK,
            init_lora_weights="gaussian",
            target_modules=["attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0", "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"],
            lora_dropout=self.LORA_DROPOUT
        )
        self.adapter_name = f"train-{uuid.uuid4().hex[:8]}"
        self.unet.add_adapter(unet_lora_config, adapter_name=self.adapter_name)
        self.unet.set_adapters([self.adapter_name])
        
        # LoRAパラメータのみ収集
        lora_params = [p for n, p in self.unet.named_parameters() if p.requires_grad]
        print(f"LoRA学習可能パラメータ数: {sum(p.numel() for p in lora_params)}")
        
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.PHASE1_LR,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )
        
        return lora_params
    
    def _setup_phase2_controlnet(self):
        """Phase 2: ControlNetのみ学習。LoRAは凍結（有効のまま）。"""
        print("\n" + "="*50)
        print("Phase 2: ControlNet学習セットアップ (低ノイズ領域)")
        print("="*50)
        
        # LoRAを凍結（アダプタは有効のまま推論に使用）
        for n, p in self.unet.named_parameters():
            p.requires_grad_(False)
        self.unet.eval()
        
        # ControlNetの浅い層のみ学習可能にする
        for c in self.controlnets:
            c.requires_grad_(False)
            c.train()
            
            # conv_in (入力層) は常に学習対象
            c.conv_in.requires_grad_(True)
            if hasattr(c, 'time_embedding'):
                c.time_embedding.requires_grad_(True)

            # down_blocks の前半のみ学習対象
            num_down = len(c.down_blocks)
            trainable_count = max(1, int(num_down * self.CONTROLNET_TRAINABLE_RATIO))
            for i in range(trainable_count):
                c.down_blocks[i].requires_grad_(True)

            # controlnet_down_blocks (ゼロ畳み込み) も対応する浅い部分のみ
            num_zero_convs = len(c.controlnet_down_blocks)
            convs_per_block = num_zero_convs // num_down if num_down > 0 else num_zero_convs
            trainable_zero_convs = trainable_count * convs_per_block
            for i in range(trainable_zero_convs):
                c.controlnet_down_blocks[i].requires_grad_(True)

            frozen = sum(1 for p in c.parameters() if not p.requires_grad)
            total = sum(1 for p in c.parameters())
            print(f"ControlNet: {total - frozen}/{total} パラメータが学習対象 "
                  f"(down_blocks 0-{trainable_count-1}/{num_down-1} を学習)")
        
        # ControlNetパラメータのみ収集
        cn_params = []
        for c in self.controlnets:
            cn_params.extend([p for p in c.parameters() if p.requires_grad])
        print(f"ControlNet学習可能パラメータ数: {sum(p.numel() for p in cn_params)}")
        
        # GradScalerをリセット
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        
        self.optimizer = torch.optim.AdamW(
            cn_params,
            lr=self.PHASE2_LR,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )
        
        return cn_params
        
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

        # テキスト埋め込みの準備
        cond_tokens = self.tokenizer([prompt], padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        uncond_tokens = self.tokenizer([""], padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            cond = self.text_encoder(cond_tokens.input_ids)[0]
            uncond = self.text_encoder(uncond_tokens.input_ids)[0]
        text_embeds = torch.cat([uncond, cond], dim=0)

        # スケジューラの準備
        scheduler = DDIMScheduler.from_pretrained(self.PRETRAINED_MODEL, subfolder="scheduler", local_files_only=True)
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

    def _train_phase1_lora(self):
        """Phase 1: LoRAのみ学習。タイムステップの前半（高t = 高ノイズ）に集中。"""
        image = self.input_image
        trainable_params = self._setup_phase1_lora()
        
        cond_tokens = self.tokenizer(
            [self.train_prompt],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        cond_emb = self.text_encoder(cond_tokens.input_ids.to(self.device))[0]
        
        num_timesteps = self.ddpm_scheduler.config.num_train_timesteps  # 1000
        prev_lpips = None
        no_improve_streak = 0
        
        progress_bar = tqdm(range(self.PHASE1_STEPS), desc="Phase1-LoRA", leave=True)
        
        for step in progress_bar:
            latent = pil_to_latent(self.vae, image, self.device, self.dtype)
            bsz = latent.shape[0]
            
            # 高tのみサンプリング: t ∈ [T_MID, num_timesteps-1]
            t = torch.randint(
                self.T_MID, num_timesteps,
                (bsz,), device=self.device, dtype=torch.long
            )
            
            noise = torch.randn_like(latent, device=self.device, dtype=self.dtype)
            noisy_latent = self.ddpm_scheduler.add_noise(latent, noise, t)
            
            # ControlNetは凍結だが順伝播は行う（構造情報を提供）
            with torch.no_grad():
                down_res, mid_res = self._apply_controlnets(
                    noisy_latent, t, cond_emb, self.control_images, strengths=[1.0] * len(self.control_images)
                )
            
            model_pred = self.unet(
                noisy_latent, t, cond_emb,
                down_block_additional_residuals=[s.to(dtype=self.dtype) for s in down_res],
                mid_block_additional_residual=mid_res.to(dtype=self.dtype),
            ).sample
            
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # # 評価と早期停止
            # if step % self.CLIP_EVAL_INTERVAL == 0:
            #     avg_dist = self._evaluate(step, latent, self.control_images, image)
                
            #     if prev_lpips is None:
            #         improvement_rate = 0.0
            #         no_improve_streak = 0
            #     else:
            #         improvement = prev_lpips - avg_dist
            #         improvement_rate = improvement / (prev_lpips + 1e-12)
                    
            #         if improvement_rate >= self.PERCEPTUAL_THRESHOLD:
            #             no_improve_streak = 0
            #         else:
            #             no_improve_streak += 1
            #             if no_improve_streak >= self.PERCEPTUAL_PATIENCE:
            #                 tqdm.write(f"Phase1 早期停止: 改善率 {improvement_rate:.4f} < {self.PERCEPTUAL_THRESHOLD} が {self.PERCEPTUAL_PATIENCE} 回連続")
            #                 break
                
            #     prev_lpips = avg_dist
            #     tqdm.write(f"\n[Phase1] LPIPS={avg_dist:.4f}, 改善率={improvement_rate*100:.2f}%")
            #     sys.stdout.flush()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        print(f"Phase 1 完了: LoRA学習済み")
    
    def _train_phase2_controlnet(self):
        """Phase 2: ControlNetのみ学習。タイムステップの後半（低t = 低ノイズ）に集中。"""
        image = self.input_image
        trainable_params = self._setup_phase2_controlnet()
        
        cond_tokens = self.tokenizer(
            [self.train_prompt],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        cond_emb = self.text_encoder(cond_tokens.input_ids.to(self.device))[0]
        
        prev_lpips = None
        no_improve_streak = 0
        
        progress_bar = tqdm(range(self.PHASE2_STEPS), desc="Phase2-ControlNet", leave=True)
        
        for step in progress_bar:
            latent = pil_to_latent(self.vae, image, self.device, self.dtype)
            bsz = latent.shape[0]
            
            # 低tのみサンプリング: t ∈ [0, T_MID)
            t = torch.randint(
                0, self.T_MID,
                (bsz,), device=self.device, dtype=torch.long
            )
            
            noise = torch.randn_like(latent, device=self.device, dtype=self.dtype)
            noisy_latent = self.ddpm_scheduler.add_noise(latent, noise, t)
            
            # ControlNet順伝播（勾配あり）
            down_res, mid_res = self._apply_controlnets(
                noisy_latent, t, cond_emb, self.control_images, strengths=[1.0] * len(self.control_images)
            )
            
            # UNet順伝播（LoRA有効だが凍結、勾配はControlNet residualsを経由して逆伝播）
            model_pred = self.unet(
                noisy_latent, t, cond_emb,
                down_block_additional_residuals=[s.to(dtype=self.dtype) for s in down_res],
                mid_block_additional_residual=mid_res.to(dtype=self.dtype),
            ).sample
            
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # # 評価と早期停止
            # if step % self.CLIP_EVAL_INTERVAL == 0:
            #     avg_dist = self._evaluate(step, latent, self.control_images, image)
                
            #     if prev_lpips is None:
            #         improvement_rate = 0.0
            #         no_improve_streak = 0
            #     else:
            #         improvement = prev_lpips - avg_dist
            #         improvement_rate = improvement / (prev_lpips + 1e-12)
                    
            #         if improvement_rate >= self.PERCEPTUAL_THRESHOLD:
            #             no_improve_streak = 0
            #         else:
            #             no_improve_streak += 1
            #             if no_improve_streak >= self.PERCEPTUAL_PATIENCE:
            #                 tqdm.write(f"Phase2 早期停止: 改善率 {improvement_rate:.4f} < {self.PERCEPTUAL_THRESHOLD} が {self.PERCEPTUAL_PATIENCE} 回連続")
            #                 break
                
            #     prev_lpips = avg_dist
            #     tqdm.write(f"\n[Phase2] LPIPS={avg_dist:.4f}, 改善率={improvement_rate*100:.2f}%")
            #     sys.stdout.flush()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        print(f"Phase 2 完了: ControlNet学習済み")

    def forward(
        self,    
        input_image: Image.Image,
        train_prompt: str,
        inference_prompt: str,
        negative_prompt: str,
        attn_word: str,
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
        
        # 1. 二段階トレーニング
        self._train_phase1_lora()
        self._train_phase2_controlnet()
        
        # 2. 推論のセットアップ
        torch.manual_seed(1234)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        for c in self.controlnets:
            c.eval()
            
        inference_tokens = self.tokenizer([inference_prompt], truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        negative_tokens = self.tokenizer([negative_prompt], truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        
        cond_emb = self.text_encoder(inference_tokens.input_ids.to(self.device))[0]
        uncond_emb = self.text_encoder(negative_tokens.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_emb, cond_emb], dim=0)
        
        # 経年変化エフェクト用のアテンションプロセッサを設定
        aging_processor = None
        if attn_word is not None:
            aging_token_indices = find_token_indices(self.tokenizer, inference_tokens.input_ids, attn_word)
            aging_processor = AgingAttentionProcessor(
                aging_token_indices,
                scale_max=self.ATTN_SCALE_MAX,
                zero_to_one=self.ATTN_ZERO_TO_ONE
            )
            self.unet.set_attn_processor(aging_processor)
        
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
            
            # # LoRAスケールをフレームに応じて段階的に低減 (1.0 → LORA_SCALE_MIN)
            # alpha = (i + 1) / num_frames
            # lora_scale = 0.3
            # self.unet.set_adapters([self.adapter_name], weights=[0.5])
            
            # 特定のタイムステップで元の画像と混合されたノイズから開始
            noisy_latent = self.ddim_scheduler.add_noise(start_latent, noise, timesteps[t_index])
            
            # 経年変化の強度を更新
            if aging_processor is not None:
                aging_processor.current_factor = math.sin(normalized_i)
            
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
