"""
wsim プロジェクトで使用される全モデルを事前ダウンロード・キャッシュするスクリプト。
GPU不要（CPU上でロードしてすぐ解放）。HuggingFace キャッシュにモデルを保存するだけ。

使い方:
    python preload_models.py                  # 全モデルをダウンロード
    python preload_models.py --groups sd15    # SD v1.5 関連のみ
    python preload_models.py --groups sdxl sd3  # SDXL と SD3 関連
    python preload_models.py --list           # グループ一覧を表示
"""

import argparse
import gc
import os
from pathlib import Path

# ============================================
# モデル定義（グループ別）
# ============================================

MODEL_GROUPS = {
    "sd15": {
        "description": "Stable Diffusion v1.5 + ControlNet (proposed, alltrain, nocontrol, notrain, linear, alltrain_control で使用)",
        "models": [
            ("diffusers", "DDIMScheduler",       "runwayml/stable-diffusion-v1-5", {"subfolder": "scheduler"}),
            ("diffusers", "DDPMScheduler",       "runwayml/stable-diffusion-v1-5", {"subfolder": "scheduler"}),
            ("transformers", "CLIPTokenizer",    "runwayml/stable-diffusion-v1-5", {"subfolder": "tokenizer"}),
            ("transformers", "CLIPTextModel",    "runwayml/stable-diffusion-v1-5", {"subfolder": "text_encoder"}),
            ("diffusers", "AutoencoderKL",       "runwayml/stable-diffusion-v1-5", {"subfolder": "vae"}),
            ("diffusers", "UNet2DConditionModel","runwayml/stable-diffusion-v1-5", {"subfolder": "unet"}),
            ("diffusers", "ControlNetModel",     "lllyasviel/sd-controlnet-canny", {}),
        ],
    },
    "sdxl": {
        "description": "Stable Diffusion XL + ControlNet (sdxl で使用)",
        "models": [
            ("diffusers", "DDIMScheduler",              "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "scheduler"}),
            ("diffusers", "DDPMScheduler",              "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "scheduler"}),
            ("transformers", "CLIPTokenizer",           "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "tokenizer"}),
            ("transformers", "CLIPTokenizer",           "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "tokenizer_2"}),
            ("transformers", "CLIPTextModel",           "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "text_encoder"}),
            ("transformers", "CLIPTextModelWithProjection", "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "text_encoder_2"}),
            ("diffusers", "AutoencoderKL",              "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "vae"}),
            ("diffusers", "UNet2DConditionModel",       "stabilityai/stable-diffusion-xl-base-1.0", {"subfolder": "unet"}),
            ("diffusers", "ControlNetModel",            "diffusers/controlnet-canny-sdxl-1.0", {}),
        ],
    },
    "sd3": {
        "description": "Stable Diffusion 3.5 Medium + ControlNet (sd3 で使用)",
        "models": [
            ("diffusers", "FlowMatchEulerDiscreteScheduler", "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "scheduler"}),
            ("transformers", "CLIPTokenizer",                "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "tokenizer"}),
            ("transformers", "CLIPTokenizer",                "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "tokenizer_2"}),
            ("transformers", "T5TokenizerFast",              "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "tokenizer_3"}),
            ("transformers", "CLIPTextModelWithProjection",  "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "text_encoder"}),
            ("transformers", "CLIPTextModelWithProjection",  "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "text_encoder_2"}),
            ("transformers", "T5EncoderModel",               "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "text_encoder_3"}),
            ("diffusers", "AutoencoderKL",                   "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "vae"}),
            ("diffusers", "SD3Transformer2DModel",           "stabilityai/stable-diffusion-3.5-medium", {"subfolder": "transformer"}),
            ("diffusers", "SD3ControlNetModel",              "InstantX/SD3-Controlnet-Canny", {}),
        ],
    },
    "depth": {
        "description": "DPT 深度推定モデル (depth_process で使用)",
        "models": [
            ("transformers", "DPTImageProcessor",       "Intel/dpt-hybrid-midas", {}),
            ("transformers", "DPTForDepthEstimation",   "Intel/dpt-hybrid-midas", {}),
        ],
    },
    "lpips": {
        "description": "LPIPS (VGG) 知覚距離モデル",
        "models": [
            # lpips は独自のダウンロード機構を使用
            ("lpips", "LPIPS", "vgg", {}),
        ],
    },
    "qwen_vlm": {
        "description": "Qwen3-VL-8B VLM (vlm.py で使用)",
        "models": [
            ("transformers", "Qwen3VLForConditionalGeneration", "Qwen/Qwen3-VL-8B-Instruct", {}),
            ("transformers", "AutoProcessor",                   "Qwen/Qwen3-VL-8B-Instruct", {}),
        ],
    },
    "mistral_vlm": {
        "description": "Mistral3-14B VLM (vlm_mistral.py で使用)",
        "models": [
            ("transformers", "MistralCommonBackend",             "mistralai/Ministral-3-14B-Instruct-2512", {}),
            ("transformers", "Mistral3ForConditionalGeneration", "mistralai/Ministral-3-14B-Instruct-2512", {}),
        ],
    },
}


# キャッシュ判定用: クラス名 → 確認すべき設定ファイル名
_CONFIG_FILE_MAP = {
    "Scheduler": "scheduler_config.json",
    "Tokenizer": "tokenizer_config.json",
    "ImageProcessor": "preprocessor_config.json",
    "Processor": "preprocessor_config.json",
}


def _get_config_filename(class_name: str) -> str:
    """クラス名からキャッシュ確認用の設定ファイル名を推定する"""
    for suffix, fname in _CONFIG_FILE_MAP.items():
        if suffix in class_name:
            return fname
    return "config.json"


def is_model_cached(package: str, class_name: str, model_id: str, kwargs: dict) -> bool:
    """モデルがすでにキャッシュされているか確認する"""
    if package == "lpips":
        # lpips はホームディレクトリ直下の .cache にキャッシュされる
        cache_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        # vgg / alex / squeeze のいずれか
        for f in cache_path.glob(f"{model_id}*"):
            if f.is_file():
                return True
        return False

    try:
        from huggingface_hub import try_to_load_from_cache

        config_name = _get_config_filename(class_name)
        subfolder = kwargs.get("subfolder")
        if subfolder:
            config_name = f"{subfolder}/{config_name}"

        result = try_to_load_from_cache(model_id, config_name)
        # result が文字列（パス）ならキャッシュ済み
        return isinstance(result, str)
    except Exception:
        return False


def load_single_model(package: str, class_name: str, model_id: str, kwargs: dict):
    """単一モデルをロード（キャッシュ目的）してすぐ解放。キャッシュ済みならスキップ。"""
    import importlib

    label = f"{class_name}({model_id}"
    if kwargs.get("subfolder"):
        label += f", subfolder={kwargs['subfolder']}"
    label += ")"

    # キャッシュ済みチェック
    if is_model_cached(package, class_name, model_id, kwargs):
        print(f"  スキップ: {label} ... ⏭ (キャッシュ済み)")
        return "skipped"

    print(f"  ロード中: {label} ...", end=" ", flush=True)

    try:
        if package == "lpips":
            import lpips
            model = lpips.LPIPS(net=model_id)
            del model
        else:
            mod = importlib.import_module(package)
            cls = getattr(mod, class_name)
            obj = cls.from_pretrained(model_id, **kwargs)
            del obj

        gc.collect()
        print("✓")
        return "success"
    except Exception as e:
        print(f"✗ ({e})")
        return "failed"


def main():
    parser = argparse.ArgumentParser(description="wsim モデル事前ダウンロード")
    parser.add_argument(
        "--groups", nargs="*", default=None,
        choices=list(MODEL_GROUPS.keys()),
        help="ダウンロードするグループ（未指定で全て）"
    )
    parser.add_argument("--list", action="store_true", help="グループ一覧を表示して終了")
    args = parser.parse_args()

    if args.list:
        print("利用可能なグループ:")
        for name, info in MODEL_GROUPS.items():
            print(f"  {name:15s} - {info['description']}")
        return

    groups = args.groups if args.groups else list(MODEL_GROUPS.keys())

    total = 0
    success = 0
    skipped = 0

    for group_name in groups:
        info = MODEL_GROUPS[group_name]
        print(f"\n{'='*60}")
        print(f"[{group_name}] {info['description']}")
        print(f"{'='*60}")

        for package, class_name, model_id, kwargs in info["models"]:
            total += 1
            result = load_single_model(package, class_name, model_id, kwargs)
            if result == "success":
                success += 1
            elif result == "skipped":
                skipped += 1

    downloaded = success
    failed = total - success - skipped
    print(f"\n{'='*60}")
    print(f"完了: 全{total}モデル — ダウンロード: {downloaded}, スキップ(キャッシュ済み): {skipped}, 失敗: {failed}")
    if failed > 0:
        print(f"警告: {failed} 個のモデルがダウンロードに失敗しました")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
