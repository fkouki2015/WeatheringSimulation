import copy
import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import ZImagePipeline
from diffusers import ZImageControlNetModel
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig
from huggingface_hub import hf_hub_download
import numpy as np

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

def canny_process(image: Image.Image, low_threshold=100, high_threshold=200):
    import cv2
    import numpy as np

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image_cv, low_threshold, high_threshold)
    edges_pil = Image.fromarray(edges).convert("RGB")
    return edges_pil


# def calculate_shift(
#     image_seq_len,
#     base_seq_len: int = 256,
#     max_seq_len: int = 4096,
#     base_shift: float = 0.5,
#     max_shift: float = 1.15,
# ):
#     m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
#     b = base_shift - m * base_seq_len
#     mu = image_seq_len * m + b
#     return mu


#  logit normal
def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1

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

        self.pipeline = ZImagePipeline.from_pretrained(
            self.PRETRAINED_MODEL,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.noise_scheduler = self.pipeline.scheduler

        # self.controlnet = ZImageControlNetModel.from_single_file(
        #     hf_hub_download(
        #         "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union",
        #         filename="Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
        #     ),
        #     torch_dtype=torch.bfloat16,
        # )
        # Share transformer modules with controlnet
        # self.controlnet = ZImageControlNetModel.from_transformer(self.controlnet, self.transformer)
        # self.controlnet.gradient_checkpointing = False

        # self.controlnet.to(self.device, self.dtype)

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        # self.controlnet.requires_grad_(False)
        
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
        # self.controlnet.add_adapter(transformer_lora_config)

        # self.controlnet.train()
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

    def train_model(self, x_0, train_prompt: str):
        self._setup_training()

        # control_image = pil_to_latent(self.vae, self.control_image, self.device, self.dtype)
        # control_image = control_image.unsqueeze(2)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=[train_prompt],
            device=self.device,
            do_classifier_free_guidance=False,
        )

        progress_bar = tqdm(range(self.TRAIN_STEPS), desc="Train", leave=True)
        for _ in progress_bar:
            z = torch.randn_like(x_0)
            t_scaler = sample_t_logit_normal(batch_size=1, device=self.device)
            t = t_scaler.view(-1, 1, 1, 1).to(self.dtype)
            x_t = (1.0 - t) * x_0 + t * z

            t_in = 1.0 - t_scaler.expand(x_t.shape[0])
            x_t_in_list = list(x_t.to(self.dtype).unsqueeze(2).unbind(dim=0))

            # controlnet_block_samples = self.controlnet(
            #     x_t_in_list, 
            #     t_in, 
            #     prompt_embeds_list,
            #     control_image,
            #     conditioning_scale=1.0,
            # )

            model_pred = self.transformer(
                x_t_in_list, 
                t_in, 
                prompt_embeds,
                # controlnet_block_samples=controlnet_block_samples
            )[0]
            model_pred = torch.stack(model_pred, dim=0) # shape: (1, C, F, H, W)
            pred_v = model_pred.squeeze(2) # shape: (1, C, H, W)
            pred_v = -pred_v

            target = z - x_0

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
        input_image = input_image.resize(self.RESOLUTION, Image.LANCZOS)

        # self.control_image = canny_process(self.input_image)

        x_0 = pil_to_latent(self.pipeline, input_image, self.device, self.dtype)

        # モデルのトレーニング
        self.train_model(x_0, train_prompt)

        torch.manual_seed(1234)
        self.vae.eval()
        self.transformer.eval()
        # self.controlnet.eval()
        self.text_encoder.eval()

        pos_prompt_embeds, neg_prompt_embeds = self.pipeline.encode_prompt(
            prompt=[inference_prompt],
            negative_prompt=[negative_prompt],
            device=self.device,
            do_classifier_free_guidance=True,
        )

        # image_seq_len = (x_0.shape[2] // 2) * (x_0.shape[3] // 2)

        # control_image = pil_to_latent(self.pipeline, control_image, self.device, self.dtype)
        # control_image = control_image.unsqueeze(2)
        # control_image_in = control_image.repeat(2, 1, 1, 1, 1)

        
        frames = []
        for i in range(num_frames):
            sigmas = np.linspace(1.0, 1 / self.INFER_STEPS, self.INFER_STEPS)
            image_seq_len = (x_0.shape[2] // 2) * (x_0.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                self.noise_scheduler.config.base_image_seq_len,
                self.noise_scheduler.config.max_image_seq_len,
                self.noise_scheduler.config.base_shift,
                self.noise_scheduler.config.max_shift,
            )
            self.noise_scheduler.set_timesteps(
                self.INFER_STEPS, 
                device=self.device, 
                sigmas=sigmas,
                mu=mu
            )
            timesteps = self.noise_scheduler.timesteps
            # t_index: 0...INFER_STEPS-1
            normalized_i = (i + 1) / (num_frames)
            t_index = int((self.INFER_STEPS - 1) * (1.0 - normalized_i)**2.0)
            t_index = max(0, min(t_index, self.INFER_STEPS - 1))

            t_scaler = timesteps[t_index] / 1000
            t = t_scaler.view(-1, 1, 1, 1).to(self.dtype)
            z = torch.randn_like(x_0)
            x_t = (1.0 - t) * x_0 + t * z

            
            # デノイズ
            progress = tqdm(range(t_index, len(timesteps)), desc=f"{i + 1}/{num_frames}")
            for index in progress:
                with torch.no_grad():
                    timestep_scaler = timesteps[index].to(self.device)
                    t_in = (1000 - timestep_scaler.expand(x_t.shape[0])) / 1000
                    t_in = t_in.repeat(2)

                    x_t_in = x_t.to(self.dtype).repeat(2, 1, 1, 1)
                    x_t_in_list = list(x_t_in.unsqueeze(2).unbind(dim=0)) # shape: (2, C, H, W) -> list of (C, F, H, W)
                    
                    prompts_in = pos_prompt_embeds + neg_prompt_embeds

                    # controlnet_block_samples = self.controlnet(
                    #     x_t_in_list, 
                    #     t_in, 
                    #     prompts_in,
                    #     control_image_in,
                    #     conditioning_scale=1.0,
                    # )
            
                    model_pred_list = self.transformer(
                        x_t_in_list, 
                        t_in, 
                        prompts_in,
                        # controlnet_block_samples=controlnet_block_samples
                    ).sample # shape: list of (C, F, H, W)
                    model_pred = torch.stack(model_pred_list, dim=0) # shape: (2, C, F, H, W)
                    model_pred = model_pred.squeeze(2) # shape: (2, C, H, W)
                    noise_pos, noise_neg = model_pred.chunk(2, dim=0) # shape: (1, C, H, W) each

                    noise_guided = noise_pos + guidance_scale * (noise_pos - noise_neg)
                    noise_guided = -noise_guided

                    x_t = self.noise_scheduler.step(noise_guided.float(), timestep_scaler, x_t).prev_sample

            output_image = latent_to_pil(self.pipeline, x_t, self.dtype)[0]
            # output_image = output_image.resize((512, 512), Image.LANCZOS)
            frames.append(output_image)

        return frames

if __name__ == "__main__":
    from PIL import Image

    model = SD3Model()
    input_image = Image.open("images_test/image_004.jpg").convert("RGB")
    train_prompt = "A camera"
    inference_prompt = "A heavily rusted camera"
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
