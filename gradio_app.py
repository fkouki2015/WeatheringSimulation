#!/usr/bin/env python3
"""
Gradio app for Weathering Model.

3-step workflow:
  Step 1: VLM でプロンプト生成
  Step 2: モデル学習
  Step 3: フレーム連続生成

Usage:
    cd /work/DDIPM/kfukushima/wsim
    python3 gradio_app.py [--port 7860] [--device cuda]
"""

import argparse
import os
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr
from PIL import Image

# weathering_model / vlm はトップレベルで import するが、
# 重いモデルは呼び出し時に初めてロードされる（遅延ロード）
sys.path.insert(0, str(Path(__file__).parent))
import vlm as vlm_module
from weathering_model import WeatheringModel


# ---------------------------------------------------------------------------
# グローバル状態（セッション共有: シングルユーザ想定）
# ---------------------------------------------------------------------------
_weathering_model: WeatheringModel | None = None
_model_lock = threading.Lock()


def _get_weathering_model(device: str) -> WeatheringModel:
    global _weathering_model
    if _weathering_model is None:
        with _model_lock:
            if _weathering_model is None:
                print("Loading WeatheringModel...")
                _weathering_model = WeatheringModel(device=device)
                print("WeatheringModel loaded.")
    return _weathering_model


# ---------------------------------------------------------------------------
# Step 1: VLM プロンプト生成
# ---------------------------------------------------------------------------
def step1_generate_prompt(image, device):
    """アップロード画像からVLMでプロンプトを生成する"""
    if image is None:
        return gr.update(), gr.update(value="⚠️ 画像をアップロードしてください")

    # PIL 画像を一時ファイルに保存（vlm は file path を受け取る）
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = Image.fromarray(image).convert("RGB")
        img.save(tmp_path)

    try:
        input_prompt, output_prompt, instruction = vlm_module.vlm_inference(
            mode="age", image_path=tmp_path
        )
        return (
            gr.update(value=input_prompt),
            gr.update(value=output_prompt),
            gr.update(value="✅ プロンプト生成完了"),
        )
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(value=f"❌ エラー: {e}"),
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Step 2: 学習
# ---------------------------------------------------------------------------
def step2_train(image, input_prompt, output_prompt, learning_rate, train_steps, lora_rank, device):
    """WeatheringModel でファインチューニングを実行する"""
    if image is None:
        yield gr.update(value="⚠️ Step 1 で画像をアップロードしてください")
        return
    if not input_prompt.strip():
        yield gr.update(value="⚠️ Input Prompt が空です")
        return

    yield gr.update(value="🔄 モデルロード中...")
    model = _get_weathering_model(device)

    # 画像の準備
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    else:
        pil_img = Image.fromarray(image).convert("RGB")

    # output_prompt を train_prompt として使う（学習プロンプト = 劣化後の状態）
    train_prompt = output_prompt.strip() if output_prompt.strip() else input_prompt.strip()

    yield gr.update(value=f"🔄 学習開始... (LR={learning_rate}, Steps={train_steps}, Rank={lora_rank})")

    try:
        # LoRA rank を設定
        model.RANK = int(lora_rank)

        # ログ収集（tqdm は stderr に出るので stdout に切り替え）
        old_stdout = sys.stdout
        log_lines = []

        class LogCapture:
            def write(self, s):
                old_stdout.write(s)
                if s.strip():
                    log_lines.append(s.strip())
            def flush(self):
                old_stdout.flush()

        sys.stdout = LogCapture()
        try:
            model.train_only(
                input_image=pil_img,
                train_prompt=train_prompt,
                learning_rate=float(learning_rate),
                train_steps=int(train_steps),
            )
        finally:
            sys.stdout = old_stdout

        yield gr.update(value="✅ 学習完了！Step 3 で生成できます\n" + "\n".join(log_lines[-10:]))

    except Exception as e:
        import traceback
        yield gr.update(value=f"❌ 学習エラー:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Step 3: フレーム生成
# ---------------------------------------------------------------------------
def step3_generate(output_prompt, negative_prompt, num_frames, guidance_scale, attn_word, device):
    """学習済みモデルで連続フレームを生成する"""
    model = _get_weathering_model(device)

    if not hasattr(model, "input_image") or model.input_image is None:
        return [], "⚠️ 先に Step 2 の学習を完了させてください"

    if not output_prompt.strip():
        return [], "⚠️ Output Prompt が空です"

    try:
        frames = model.generate_frames(
            inference_prompt=output_prompt.strip(),
            negative_prompt=negative_prompt.strip(),
            attn_word=attn_word.strip() if attn_word.strip() else None,
            guidance_scale=float(guidance_scale),
            num_frames=int(num_frames),
        )
        return frames, f"✅ {len(frames)} フレーム生成完了"
    except Exception as e:
        import traceback
        return [], f"❌ 生成エラー:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui(device: str):
    css = """
    .tab-header { font-size: 1.1rem; font-weight: 700; }
    .status-box { font-size: 0.85rem; }
    """

    with gr.Blocks(title="Weathering Model", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚙️ Weathering Model")
        gr.Markdown("3ステップで画像の経年変化を生成します。")

        # 共有ステート
        shared_image = gr.State(None)
        shared_input_prompt = gr.State("")
        shared_output_prompt = gr.State("")

        with gr.Tabs():

            # ====== Tab 1: プロンプト生成 ======
            with gr.Tab("Step 1: プロンプト生成"):
                gr.Markdown("### 入力画像をアップロードしてVLMでプロンプトを自動生成します")
                with gr.Row():
                    with gr.Column(scale=1):
                        t1_image = gr.Image(label="入力画像", type="numpy")
                        t1_btn = gr.Button("🤖 VLM でプロンプト生成", variant="primary")
                    with gr.Column(scale=1):
                        t1_input_prompt = gr.Textbox(
                            label="Input Prompt（編集可）",
                            placeholder="例: A clean car",
                            lines=2,
                        )
                        t1_status = gr.Textbox(label="ステータス", interactive=False, lines=1, elem_classes="status-box")

                t1_btn.click(
                    fn=step1_generate_prompt,
                    inputs=[t1_image, gr.State(device)],
                    outputs=[t1_input_prompt, shared_output_prompt, t1_status],
                )

            # ====== Tab 2: 学習 ======
            with gr.Tab("Step 2: 学習"):
                gr.Markdown("### 入力画像と生成済みプロンプトを使ってLoRAで学習します")
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_image = gr.Image(label="入力画像（Step 1 と同じ）", type="numpy")
                        t2_input_prompt = gr.Textbox(
                            label="Input Prompt（学習用）",
                            placeholder="Step 1 から自動でコピー、または直接入力",
                            lines=2,
                        )
                        t2_output_prompt = gr.Textbox(
                            label="Output Prompt（学習ターゲット）",
                            placeholder="Step 1 から自動でコピー、または直接入力",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        t2_lr = gr.Number(label="Learning Rate", value=1e-5, precision=8)
                        t2_steps = gr.Slider(label="Train Steps", minimum=50, maximum=1000, step=50, value=450)
                        t2_rank = gr.Slider(label="LoRA Rank", minimum=2, maximum=64, step=2, value=8)
                        t2_btn = gr.Button("🚀 学習開始", variant="primary")
                        t2_log = gr.Textbox(label="学習ログ", interactive=False, lines=8, elem_classes="status-box")

                t2_btn.click(
                    fn=step2_train,
                    inputs=[t2_image, t2_input_prompt, t2_output_prompt, t2_lr, t2_steps, t2_rank, gr.State(device)],
                    outputs=[t2_log],
                )

                # Step 1 → Step 2 への値引き継ぎボタン
                with gr.Row():
                    sync_btn = gr.Button("↩ Step 1 の画像・プロンプトを引き継ぐ")

                def _sync_from_step1(img, inp, out):
                    return img, inp, out

                sync_btn.click(
                    fn=_sync_from_step1,
                    inputs=[t1_image, t1_input_prompt, shared_output_prompt],
                    outputs=[t2_image, t2_input_prompt, t2_output_prompt],
                )

            # ====== Tab 3: 生成 ======
            with gr.Tab("Step 3: 生成"):
                gr.Markdown("### 学習済みLoRAで連続フレームを生成します")
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_output_prompt = gr.Textbox(
                            label="Output Prompt（生成プロンプト）",
                            placeholder="例: A heavily rusted car",
                            lines=2,
                        )
                        t3_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="（任意）",
                            lines=2,
                            value="",
                        )
                        t3_attn_word = gr.Textbox(
                            label="Aging Attention Word（任意）",
                            placeholder="例: rusted",
                            lines=1,
                            value="",
                        )
                        t3_num_frames = gr.Slider(label="Num Frames", minimum=1, maximum=20, step=1, value=5)
                        t3_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.5, value=7.5)
                        t3_btn = gr.Button("🎞️ フレーム生成", variant="primary")
                        t3_status = gr.Textbox(label="ステータス", interactive=False, lines=1, elem_classes="status-box")
                    with gr.Column(scale=2):
                        t3_gallery = gr.Gallery(label="生成フレーム", columns=5, height="auto")

                # Step 2 → Step 3 への値引き継ぎボタン
                with gr.Row():
                    sync_btn3 = gr.Button("↩ Step 2 の Output Prompt を引き継ぐ")

                sync_btn3.click(
                    fn=lambda x: x,
                    inputs=[t2_output_prompt],
                    outputs=[t3_output_prompt],
                )

                t3_btn.click(
                    fn=step3_generate,
                    inputs=[t3_output_prompt, t3_negative_prompt, t3_num_frames, t3_guidance, t3_attn_word, gr.State(device)],
                    outputs=[t3_gallery, t3_status],
                )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(device=args.device)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
