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
    LEARNING_RATE = 1e-5
    TRAIN_STEPS = 550
    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_PATH_CANNY = "lllyasviel/sd-controlnet-canny"
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
    # CONTROLNET_TRAINABLE_RATIO = 0.7

    def _setup_training(self):
        """トレーニング用にLoRAとオプティマイザを設定"""
        self.unet.requires_grad_(False)
        self.unet.train()

        for c in self.controlnets:
            # まず全体を凍結
            c.requires_grad_(True)

            # # 浅い層のみ学習可能にする
            # # conv_in (入力層) は常に学習対象
            # c.conv_in.requires_grad_(True)
            # if hasattr(c, 'time_embedding'):
            #     c.time_embedding.requires_grad_(True)

            # # down_blocks の前半のみ学習対象
            # num_down = len(c.down_blocks)
            # trainable_count = max(1, int(num_down * self.CONTROLNET_TRAINABLE_RATIO))
            # for i in range(trainable_count):
            #     c.down_blocks[i].requires_grad_(True)

            # # controlnet_down_blocks (ゼロ畳み込み) も対応する浅い部分のみ
            # # SD1.5 ControlNet: down_block 0 → controlnet_down_blocks 0,1,2
            # #                   down_block 1 → controlnet_down_blocks 3,4,5
            # #                   down_block 2 → controlnet_down_blocks 6,7,8
            # #                   down_block 3 → controlnet_down_blocks 9,10,11
            # num_zero_convs = len(c.controlnet_down_blocks)
            # convs_per_block = num_zero_convs // num_down if num_down > 0 else num_zero_convs
            # trainable_zero_convs = trainable_count * convs_per_block
            # for i in range(trainable_zero_convs):
            #     c.controlnet_down_blocks[i].requires_grad_(True)

            # # mid_block, controlnet_mid_block は凍結のまま (深い層)

            # frozen = sum(1 for p in c.parameters() if not p.requires_grad)
            # total = sum(1 for p in c.parameters())
            # print(f"ControlNet: {total - frozen}/{total} パラメータが学習対象 "
            #       f"(down_blocks 0-{trainable_count-1}/{num_down-1} を学習)")
        
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

    def train_model(self):
        """LoRAを使用してモデルをファインチューニング"""
        image = self.input_image
        
        trainable_params = self._setup_training()
        
        cond_tokens = self.tokenizer(
            [self.train_prompt],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        cond_emb = self.text_encoder(cond_tokens.input_ids.to(self.device))[0]
        
        timesteps = self.ddpm_scheduler.timesteps
        prev_lpips = None
        no_improve_streak = 0
        
        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        
        for step in progress_bar:
            # 1. 入力の準備
            latent = pil_to_latent(self.vae, image, self.device, self.dtype)
            bsz = latent.shape[0]
            # Logit normal sampling: 中間タイムステップに重点を置くサンプリング
            u = torch.sigmoid(torch.randn((bsz,), device=self.device) * 1.0 + 0.0)  # logit_std=1.0, logit_mean=0.0
            t = (u * (len(timesteps) - 1)).long().clamp(0, len(timesteps) - 2)
            
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

    def train_only(
        self,
        input_image: Image.Image,
        train_prompt: str,
        learning_rate: float = None,
        train_steps: int = None,
    ) -> None:
        """Step 1: LoRAを使用してモデルをファインチューニング（学習のみ）"""
        # パラメータ上書き
        if learning_rate is not None:
            self.LEARNING_RATE = learning_rate
        if train_steps is not None:
            self.TRAIN_STEPS = train_steps

        # パラメータを保存（RESOLUTION にリサイズして保存）
        w, h = self.RESOLUTION
        self.input_image = input_image.resize((w, h), Image.LANCZOS)
        self.train_prompt = train_prompt
        # 制御画像を前処理（リサイズ済み画像から生成）
        self.control_images = []
        canny_image = canny_process(self.input_image, self.device, self.dtype)
        self.control_images.append(canny_image)

        # 学習実行
        self.train_model()

    def generate_frames(
        self,
        inference_prompt: str,
        negative_prompt: str,
        attn_word: str,
        guidance_scale: float,
        num_frames: int,
    ) -> list[Image.Image]:
        """Step 2: 学習済みLoRAを使って連続フレームを生成"""
        # 推論のセットアップ
        torch.manual_seed(1234)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        for c in self.controlnets:
            c.eval()

        inference_tokens = self.tokenizer(
            [inference_prompt], truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        negative_tokens = self.tokenizer(
            [negative_prompt], truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )

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
        start_latent = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)
        noise = torch.randn_like(start_latent, device=self.device, dtype=self.dtype)

        frames = []

        # 生成ループ (フレームごと)
        for i in range(num_frames):
            # 経年変化係数を計算 (n=3 べき乗カーブ)
            normalized_i = (i + 1) / num_frames
            t_norm = 1.0 - (1.0 - normalized_i) ** 3
            t_index = self.INFER_STEPS - 1 - int(t_norm * (self.INFER_STEPS - 1))
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))

            noisy_latent = self.ddim_scheduler.add_noise(start_latent, noise, timesteps[t_index])

            if aging_processor is not None:
                aging_processor.current_factor = t_norm

            # デノイズ
            with torch.no_grad():
                latents = noisy_latent
                for t in tqdm(timesteps[t_index:], desc=f"{i+1}/{num_frames}"):
                    with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda"), dtype=self.dtype):

                        lat_in = torch.cat([latents] * 2, dim=0)
                        lat_in = self.ddim_scheduler.scale_model_input(lat_in, t)

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

                        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                        latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample

                output_image = latent_to_pil(self.vae, latents)[0]
                frames.append(output_image)

        return frames

    def forward(
        self,
        input_image: Image.Image,
        train_prompt: str,
        inference_prompt: str,
        negative_prompt: str,
        attn_word: str,
        guidance_scale: float,
        num_frames: int,
        learning_rate: float = None,
        train_steps: int = None,
    ) -> list[Image.Image]:
        """後方互換ラッパー: train_only() + generate_frames() を順に実行"""
        self.train_only(input_image, train_prompt, learning_rate=learning_rate, train_steps=train_steps)
        return self.generate_frames(inference_prompt, negative_prompt, attn_word, guidance_scale, num_frames)

