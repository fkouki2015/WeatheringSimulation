import copy
import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL, ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig


def pil_to_latent(vae, pil: Image.Image, device, dtype):
    x = transforms.ToTensor()(pil).unsqueeze(0).to(device)
    x = (x * 2 - 1).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(x).latent_dist.mean

    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.to(dtype=dtype)


def latent_to_pil(vae, latents):
    latents_decode = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    latents_decode = latents_decode.to(dtype=vae.dtype)

    with torch.no_grad():
        imgs = vae.decode(latents_decode, return_dict=False)[0]

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = (imgs * 255).round().to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in imgs]


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


#  logit normal
def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1

def encode_prompts(text_encoder, tokenizer, prompts, device, max_sequence_length=256):
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

    embeds = []
    for i in range(len(hidden)):
        embeds.append(hidden[i][masks[i]])
    return embeds

class SD3Model(nn.Module):
    RESOLUTION = (1024, 1024)
    LEARNING_RATE = 1e-4
    TRAIN_STEPS = 500
    DEVICE = "cuda"
    PRETRAINED_MODEL = "Tongyi-MAI/Z-Image"
    INFER_STEPS = 50
    NOISE_RATIO = 0.6
    MAX_SEQUENCE_LENGTH = 256

    def __init__(self, device: str = None):
        super().__init__()
        self.device = torch.device(device if device else self.DEVICE)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.transformer = ZImageTransformer2DModel.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="transformer",
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="vae",
        )
        self.text_encoder = AutoModel.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="text_encoder",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="tokenizer",
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.PRETRAINED_MODEL,
            subfolder="scheduler",
        )

        self.transformer.to(self.device, self.dtype)
        self.vae.to(self.device, self.dtype)
        self.text_encoder.to(self.device, self.dtype)

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
    def _setup_training(self):
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # transformer_lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=8,
        #     init_lora_weights="gaussian",
        #     target_modules=[
        #         "attention.to_out.0", "attention.to_v", "attention.to_q", "attention.to_k"
        #     ]
        # )
        # self.transformer.add_adapter(transformer_lora_config)

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

        # for n, p in self.controlnet.named_parameters():
        #     if p.requires_grad:
        #         _add_param(p, 1.0)
        
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

        prompt_embeds_list = encode_prompts(
            self.text_encoder,
            self.tokenizer,
            [self.train_prompt],
            self.device,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for _ in progress_bar:
            z = torch.randn_like(x_0)
            t_float = sample_t_logit_normal(batch_size=x_0.shape[0], device=x_0.device)
            t = t_float.view(-1, 1, 1, 1).to(self.dtype)
            x_t = (1.0 - t) * x_0 + t * z

            t_in = 1.0 - t_float
            x_t_in_list = list(x_t.to(self.dtype).unsqueeze(2).unbind(dim=0))

            pred_out = self.transformer(x_t_in_list, t_in, prompt_embeds_list).sample
            pred_v = -torch.stack([p.squeeze(1) for p in pred_out], dim=0)

            target = z - x_0
            # pred_x = pred_v * (-t) + x_t
            loss = torch.nn.functional.mse_loss(pred_v.float(), target.float())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})

        self.transformer.requires_grad_(False)

    # def _prepare_scheduler(self, latents):
    #     scheduler = copy.deepcopy(self.noise_scheduler)
    #     image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    #     mu = calculate_shift(
    #         image_seq_len,
    #         scheduler.config.get("base_image_seq_len", 256),
    #         scheduler.config.get("max_image_seq_len", 4096),
    #         scheduler.config.get("base_shift", 0.5),
    #         scheduler.config.get("max_shift", 1.15),
    #     )
    #     scheduler.sigma_min = 0.0
    #     scheduler.set_timesteps(self.INFER_STEPS, device=self.device, mu=mu)
    #     return scheduler

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

        # モデルのトレーニング
        self.train_model()

        torch.manual_seed(1234)
        self.vae.eval()
        self.transformer.eval()
        # self.controlnet.eval()
        self.text_encoder.eval()

        pos_prompt = encode_prompts(
            self.text_encoder,
            self.tokenizer,
            [inference_prompt],
            self.device,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )
        neg_prompt = encode_prompts(
            self.text_encoder,
            self.tokenizer,
            [negative_prompt],
            self.device,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )

        x_0 = pil_to_latent(self.vae, self.input_image, self.device, self.dtype)
        frames = []
        image_seq_len = (x_0.shape[2] // 2) * (x_0.shape[3] // 2)

        for i in range(num_frames):
            self.noise_scheduler.set_timesteps(self.INFER_STEPS, device=self.device)
            timesteps = self.noise_scheduler.timesteps

            # t_index: 0...INFER_STEPS-1
            normalized_i = i / max(1, num_frames - 1)
            t_index = self.INFER_STEPS - int((self.INFER_STEPS - 1) * (normalized_i ** self.NOISE_RATIO)) - 1
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))
            if num_frames == 1:
                t_index = 0

            t_float = timesteps[t_index] / self.noise_scheduler.num_train_timesteps
            t = t_float.view(-1, 1, 1, 1).to(self.dtype)
            z = torch.randn_like(x_0)
            x_t = (1.0 - t) * x_0 + t * z

            # デノイズ
            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i + 1}/{num_frames}")
            for index in progress:
                with torch.no_grad():
                    timestep = timesteps[index].to(self.device)
                    t_in = (self.noise_scheduler.num_train_timesteps - timestep) / self.noise_scheduler.num_train_timesteps
                    t_in = t_in.repeat(2)
                    
                    x_t_in = x_t.to(self.dtype).repeat(2, 1, 1, 1)
                    x_t_in_list = list(x_t_in.unsqueeze(2).unbind(dim=0)) # shape: (2, C, H, W) -> list of (C, F, H, W)
                    
                    prompts_in = pos_prompt + neg_prompt
            
                    pred_out = self.transformer(x_t_in_list, t_in, prompts_in).sample # shape: (C, F, H, W)

                    pred_v = torch.stack([p.squeeze(1) for p in pred_out], dim=0) # shape: (2, C, H, W)
                    noise_pos, noise_neg = pred_v.chunk(2, dim=0)
                    noise_guided = noise_pos + guidance_scale * (noise_pos - noise_neg)
                    noise_guided = -noise_guided

                    x_t = self.noise_scheduler.step(noise_guided.to(torch.float32), timestep, x_t).prev_sample

            output_image = latent_to_pil(self.vae, x_t)[0]
            # output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)

        return frames

if __name__ == "__main__":
    from PIL import Image

    model = SD3Model()
    input_image = Image.open("images_test/image_012.jpg").convert("RGB")
    train_prompt = "A car"
    inference_prompt = "A heavily rusted car"
    negative_prompt = ""
    guidance_scale = 7
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
