import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL, ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer

from huggingface_hub import snapshot_download


def pil_to_latent(vae, pil: Image.Image, device, dtype):
    to_tensor = transforms.ToTensor()
    x = to_tensor(pil).unsqueeze(0).to(device)
    x = (x * 2 - 1).to(dtype=torch.float32)

    with torch.no_grad():
        moments = vae.encoder(x)
        if vae.quant_conv is not None:
            moments = vae.quant_conv(moments)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
        latents = mean + std * torch.randn_like(std)

    latents = (latents - (getattr(vae.config, "shift_factor", 0.0) or 0.0)) * vae.config.scaling_factor
    return latents.to(dtype=dtype)


def latent_to_pil(vae, latents):
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents_decode = (latents / vae.config.scaling_factor) + shift
    latents_decode = latents_decode.to(dtype=vae.dtype)

    with torch.no_grad():
        imgs = vae.decode(latents_decode, return_dict=False)[0]

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = (imgs * 255).round().to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in imgs]


def get_sigmas(scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(device)]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _encode_prompt_list(text_encoder, tokenizer, prompts, device, max_sequence_length=256):
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted.append(formatted_prompt)

    text_inputs = tokenizer(
        formatted,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    masks = text_inputs.attention_mask.to(device).bool()

    with torch.no_grad():
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    embeds = [hidden[i][masks[i]] for i in range(hidden.shape[0])]
    return embeds


def _predict_noise(transformer, latents, timesteps_01, prompt_embeds_list):
    latent_in = latents.unsqueeze(2)
    latent_in_list = list(latent_in.unbind(dim=0))
    out_list = transformer(latent_in_list, timesteps_01, prompt_embeds_list)[0]
    noise_pred = -torch.stack([o.float() for o in out_list], dim=0).squeeze(2)
    return noise_pred


class SD3Model(nn.Module):
    RESOLUTION = (1024, 1024)
    LEARNING_RATE = 1e-5
    TRAIN_STEPS = 600
    DEVICE = "cuda"
    PRETRAINED_MODEL = "Tongyi-MAI/Z-Image"
    INFER_STEPS = 50
    NOISE_RATIO = 0.6
    MAX_SEQUENCE_LENGTH = 256

    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._init_models()

    def _init_models(self):

        self.transformer = ZImageTransformer2DModel.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="transformer",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.text_encoder = AutoModel.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="tokenizer",
            local_files_only=True,
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="scheduler",
            local_files_only=True,
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def _setup_training(self):
        self.transformer.train()
        self.transformer.requires_grad_(True)

        params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )
        return params

    def train_model(self):
        latent = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)
        self._setup_training()

        prompt_embeds_list = _encode_prompt_list(
            self.text_encoder,
            self.tokenizer,
            [self.train_prompt],
            self.device,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for _ in progress_bar:
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

            sigmas = get_sigmas(
                self.noise_scheduler_copy,
                timesteps,
                n_dim=latent.ndim,
                dtype=latent.dtype,
                device=latent.device,
            )
            noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

            timestep_01 = (1000 - timesteps.float()) / 1000
            model_pred = _predict_noise(self.transformer, noisy_latent, timestep_01, prompt_embeds_list)

            weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - noise.float()) ** 2).reshape(noise.shape[0], -1), 1
            ).mean()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})

        self.transformer.eval()
        self.transformer.requires_grad_(False)

    def _prepare_scheduler(self, latents):
        scheduler = copy.deepcopy(self.noise_scheduler)
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.sigma_min = 0.0
        scheduler.set_timesteps(self.INFER_STEPS, device=self.device, mu=mu)
        return scheduler

    @torch.no_grad()
    def forward(
        self,
        input_image: Image.Image,
        train_prompt: str,
        inference_prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_frames: int,
    ) -> list[Image.Image]:
        self.input_image = input_image.resize(self.RESOLUTION, Image.LANCZOS)
        self.train_prompt = train_prompt

        self.train_model()

        self.vae.eval()
        self.text_encoder.eval()

        pos_prompt = _encode_prompt_list(
            self.text_encoder,
            self.tokenizer,
            [inference_prompt],
            self.device,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )
        neg_prompt = _encode_prompt_list(
            self.text_encoder,
            self.tokenizer,
            [negative_prompt],
            self.device,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )

        base_latent = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)
        frames = []

        for i in range(num_frames):
            torch.manual_seed(1234)
            scheduler = self._prepare_scheduler(base_latent)
            timesteps = scheduler.timesteps

            normalized_i = i / max(1, num_frames - 1)
            t_index = self.INFER_STEPS - int((self.INFER_STEPS - 1) * (normalized_i ** self.NOISE_RATIO)) - 1
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            if num_frames == 1:
                t_index = 0

            sigma = get_sigmas(
                scheduler,
                timesteps[t_index],
                n_dim=base_latent.ndim,
                dtype=base_latent.dtype,
                device=base_latent.device,
            )
            noise = torch.randn_like(base_latent)
            latents = (1.0 - sigma) * base_latent + sigma * noise

            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i + 1}/{num_frames}")
            for index in progress:
                t = timesteps[index].to(self.device)
                t_01 = ((1000 - t.float()) / 1000).unsqueeze(0)

                if guidance_scale > 1.0:
                    latents_in = latents.repeat(2, 1, 1, 1)
                    prompts_in = pos_prompt + neg_prompt
                    t_in = t_01.repeat(2)
                    noise_all = _predict_noise(self.transformer, latents_in, t_in, prompts_in)
                    noise_pos, noise_neg = noise_all.chunk(2, dim=0)
                    noise_guided = noise_pos + guidance_scale * (noise_pos - noise_neg)
                else:
                    noise_guided = _predict_noise(self.transformer, latents, t_01, pos_prompt)

                latents = scheduler.step(noise_guided.to(torch.float32), t, latents, return_dict=False)[0]

            output_image = latent_to_pil(self.vae, latents)[0]
            output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)

        return frames

if __name__ == "__main__":
    from PIL import Image

    model = SD3Model()
    input_image = Image.open("/work/DDIPM/kfukushima/wsim/images_test/image_004.jpg").convert("RGB")
    train_prompt = "A camera."
    inference_prompt = "A rusted camera."
    negative_prompt = ""
    guidance_scale = 7.5
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