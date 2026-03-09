import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline
import os
import datetime
from peft import LoraConfig


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def pil_to_latent(pipeline: FluxPipeline, pil: Image.Image, device: torch.device, dtype: torch.dtype):
    image = pipeline.image_processor.preprocess(pil).to(device=device, dtype=dtype)
    with torch.inference_mode():
        latent = pipeline.vae.encode(image).latent_dist.mode()
    latent = (latent - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latent.to(device=device, dtype=dtype)


def latent_to_pil(pipeline: FluxPipeline, latent: torch.Tensor, dtype: torch.dtype):
    latent = latent.to(device=latent.device, dtype=dtype)
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    with torch.inference_mode():
        image = pipeline.vae.decode(latent, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")

def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1


class SD3Model(nn.Module):
    RESOLUTION = (1024, 1024)
    PRETRAINED_MODEL = "black-forest-labs/FLUX.1-dev"
    DEVICE = "cuda"
    TRAIN_STEPS = 1000
    LEARNING_RATE = 1e-4
    INFER_STEPS = 20
    RANK = 256

    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.bfloat16

        self.pipeline = FluxPipeline.from_pretrained(
            self.PRETRAINED_MODEL,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.noise_scheduler = self.pipeline.scheduler
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder_one = self.pipeline.text_encoder
        self.text_encoder_two = self.pipeline.text_encoder_2

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)

    def _setup_training(self):
        self.transformer.train()
        # self.transformer.requires_grad_(True)
        transformer_lora_config = LoraConfig(
            r=self.RANK,
            lora_alpha=self.RANK,
            init_lora_weights="gaussian",
            target_modules=[
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
                print(n)
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

    def train_model(self, x_0_packed, latent_image_ids, train_prompt):
        self._setup_training()

        prompt_embeds, pooled_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=train_prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
        )
        batch_size = x_0_packed.shape[0]
        guidance = torch.tensor([2.0], device=self.device, dtype=torch.float32).expand(batch_size)

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for _ in progress_bar:
            z = torch.randn_like(x_0_packed)
            t_scaler = sample_t_logit_normal(batch_size=batch_size, device=self.device)
            t = t_scaler.view(-1, 1, 1).to(self.dtype)
            x_t_packed = (1.0 - t) * x_0_packed + t * z # [B, S, D]

            model_pred = self.transformer(
                hidden_states=x_t_packed,
                timestep=t_scaler.expand(batch_size),
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                pooled_projections=pooled_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            target = z - x_0_packed
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
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
        input_image = input_image.resize(self.RESOLUTION, Image.LANCZOS)

        x_0 = pil_to_latent(self.pipeline, input_image, self.device, self.dtype)
        batch_size, latent_channels, latent_h, latent_w = x_0.shape

        x_0, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=latent_channels,
            height=self.RESOLUTION[1],
            width=self.RESOLUTION[0],
            dtype=self.dtype,
            device=self.device,
            generator=None,
            latents=x_0,
        )

        x_0_packed = self.pipeline._pack_latents(x_0, batch_size, latent_channels, latent_h, latent_w)

        # モデル学習
        self.train_model(x_0_packed, latent_image_ids, train_prompt)

        torch.manual_seed(1234)
        self.vae.eval()
        self.transformer.eval()
        self.text_encoder_one.eval()
        self.text_encoder_two.eval()

        pos_prompt_embeds, pos_pooled_embeds, pos_text_ids = self.pipeline.encode_prompt(
            prompt=inference_prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
        )
        neg_prompt_embeds, neg_pooled_embeds, neg_text_ids = self.pipeline.encode_prompt(
            prompt=negative_prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
        )

        guidance = torch.tensor([guidance_scale], device=self.device, dtype=torch.float32).expand(batch_size)

        frames = []
        for i in range(num_frames):

            sigmas = np.linspace(1.0, 1 / self.INFER_STEPS, self.INFER_STEPS)
            mu = calculate_shift(
                x_0_packed.shape[1],
                self.noise_scheduler.config.base_image_seq_len,
                self.noise_scheduler.config.max_image_seq_len,
                self.noise_scheduler.config.base_shift,
                self.noise_scheduler.config.max_shift,
            )
            self.noise_scheduler.set_timesteps(
                self.INFER_STEPS,
                device=self.device,
                sigmas=sigmas,
                mu=mu,
            )
            timesteps = self.noise_scheduler.timesteps

            normalized_i = (i + 1) / num_frames
            t_index = int((self.INFER_STEPS - 1) * (1.0 - normalized_i) ** 2.0)
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))

            t_float = timesteps[t_index] / self.noise_scheduler.num_train_timesteps
            t = t_float.view(1, 1, 1).to(dtype=self.dtype, device=self.device)
            z = torch.randn_like(x_0_packed)
            x_t_packed = (1.0 - t) * x_0_packed + t * z

            for index in tqdm(range(t_index, len(timesteps)), desc=f"{i+1}/{num_frames}"):
                timestep_scalar = timesteps[index].to(self.device)
                timestep_batch = timestep_scalar.expand(x_t_packed.shape[0]).to(dtype=self.dtype)

                with torch.no_grad():
                    with self.pipeline.transformer.cache_context("cond"):
                        pos_noise_pred = self.pipeline.transformer(
                            hidden_states=x_t_packed,
                            timestep=timestep_batch / 1000,
                            guidance=guidance,
                            encoder_hidden_states=pos_prompt_embeds,
                            txt_ids=pos_text_ids,
                            img_ids=latent_image_ids,
                            pooled_projections=pos_pooled_embeds,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                    with self.pipeline.transformer.cache_context("uncond"):
                        neg_noise_pred = self.pipeline.transformer(
                            hidden_states=x_t_packed,
                            timestep=timestep_batch / 1000,
                            guidance=guidance,
                            encoder_hidden_states=neg_prompt_embeds,
                            txt_ids=neg_text_ids,
                            img_ids=latent_image_ids,
                            pooled_projections=neg_pooled_embeds,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                    noise_guided = neg_noise_pred + guidance_scale * (pos_noise_pred - neg_noise_pred)
                    x_t_packed = self.noise_scheduler.step(noise_guided, timestep_scalar, x_t_packed).prev_sample

            x_t = self.pipeline._unpack_latents(
                x_t_packed,
                self.RESOLUTION[0],
                self.RESOLUTION[1],
                self.pipeline.vae_scale_factor,
            )
            output_image = latent_to_pil(self.pipeline, x_t, self.dtype)[0]
            # output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)

        return frames


if __name__ == "__main__":
    if not os.path.exists("temp_saves"):
        os.makedirs("temp_saves")
    model = SD3Model()
    input_image = Image.open("images_test/image_012.jpg").convert("RGB")

    frames = model(
        input_image=input_image,
        train_prompt="A car",
        inference_prompt="A car",
        negative_prompt="",
        guidance_scale=2,
        num_frames=5,
    )

    date = datetime.datetime.now().strftime("%m%d_%H%M")
    for idx, frame in enumerate(frames):
        frame.save(f"temp_saves/flux_{date}_{idx}.png")
