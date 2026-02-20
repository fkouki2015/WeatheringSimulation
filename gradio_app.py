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
import queue
import sys
import tempfile
import threading
import traceback
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
        # 7つの出力に対応した空のupdateを返す
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value="⚠️ 画像をアップロードしてください")

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
        # vlm_module.unload_vlm()
        return (
            gr.update(value=input_prompt),   # t1_input_prompt
            gr.update(value=output_prompt),  # shared_output_prompt
            gr.update(value=image),          # t2_image（Step2に自動反映）
            gr.update(value=input_prompt),   # t2_input_prompt（Step2に自動反映）
            gr.update(value=output_prompt),  # t1_output_prompt（Step1に表示）
            gr.update(value=output_prompt),  # t3_output_prompt（Step3に自動反映）
            gr.update(value="✅ プロンプト生成完了"),
        )
    except Exception as e:
        # vlm_module.unload_vlm()
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=f"❌ エラー: {e}"),
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Step 2: 学習
# ---------------------------------------------------------------------------
def step2_train(image, input_prompt, output_prompt, learning_rate, train_steps, lora_rank, use_early_stopping, device,
                progress=gr.Progress()):
    """WeatheringModel でファインチューニングを実行する"""
    if image is None:
        yield gr.update(value="⚠️ Step 1 で画像をアップロードしてください")
        return
    if not input_prompt.strip():
        yield gr.update(value="⚠️ Input Prompt が空です")
        return

    progress(0, desc="モデルロード中...")
    yield gr.update(value="🔄 モデルロード中...")
    model = _get_weathering_model(device)

    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    else:
        pil_img = Image.fromarray(image).convert("RGB")

    train_prompt = input_prompt.strip()
    total_steps = int(train_steps)

    progress(0, desc=f"学習準備中... (Steps={total_steps}, 早期停止={'ON' if use_early_stopping else 'OFF'})")
    yield gr.update(value=f"🔄 学習開始... (LR={learning_rate}, Steps={total_steps}, Rank={lora_rank}, 早期停止={'ON' if use_early_stopping else 'OFF'})")

    log_queue = queue.Queue()
    train_done = threading.Event()
    train_error = [None]

    def progress_callback(step, loss_val, total):
        # ("prog", ...) はプログレスバー更新用
        log_queue.put(("prog", step, total, loss_val))

    class LogCapture:
        def write(self, s):
            sys.__stdout__.write(s)
            if s.strip():
                # ("log", ...) はテキストログ表示用
                log_queue.put(("log", s.rstrip()))
        def flush(self):
            sys.__stdout__.flush()

    def train_thread():
        old_stdout = sys.stdout
        sys.stdout = LogCapture()
        try:
            model.RANK = int(lora_rank)
            model.train_only(
                input_image=pil_img,
                train_prompt=train_prompt,
                inference_prompt=output_prompt.strip(),  # 追加: 評価用プロンプト
                learning_rate=float(learning_rate),
                train_steps=total_steps,
                use_early_stopping=bool(use_early_stopping),
                progress_callback=progress_callback,
            )
        except Exception:
            train_error[0] = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            train_done.set()

    t = threading.Thread(target=train_thread, daemon=True)
    t.start()

    log_lines = []
    while not train_done.is_set() or not log_queue.empty():
        try:
            msg = log_queue.get(timeout=0.3)
            # メッセージタイプで分岐
            if isinstance(msg, tuple) and msg[0] == "prog":
                _, step, total, loss_val = msg
                progress(step / total, desc=f"学習中 [{step}/{total}] Loss: {loss_val:.5f}")
            elif isinstance(msg, tuple) and msg[0] == "log":
                log_lines.append(msg[1])
                yield gr.update(value="\n".join(log_lines[-20:]))  # 最新20行を表示
            else:
                # 念のため旧形式(文字列)も対応
                log_lines.append(str(msg))
                yield gr.update(value="\n".join(log_lines[-20:]))
        except queue.Empty:
            pass
    t.join()

    if train_error[0]:
        yield gr.update(value=f"❌ 学習エラー:\n{train_error[0]}")
    else:
        yield gr.update(value="✅ 学習完了！Step 3 で生成できます\n" + "\n".join(log_lines[-30:]))


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
    .status-box {
        font-size: 0.85rem;
    }
    """

    with gr.Blocks(title="Weathering Simulation", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Weathering Simulation")
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
                        t1_output_prompt = gr.Textbox(
                            label="Output Prompt（編集可）",
                            placeholder="例: A heavily rusted car",
                            lines=2,
                        )
                        t1_status = gr.Textbox(label="ステータス", interactive=False, lines=2, elem_classes="status-box")

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
                    with gr.Column(scale=1):
                        t2_lr = gr.Number(label="Learning Rate", value=1e-5, precision=8)
                        t2_steps = gr.Slider(label="Max Train Steps", minimum=50, maximum=1000, step=50, value=450)
                        t2_rank = gr.Slider(label="LoRA Rank", minimum=2, maximum=64, step=2, value=8)
                        t2_early_stop = gr.Checkbox(label="早期停止を使用（LPIPS評価）", value=True)
                        t2_btn = gr.Button("🚀 学習開始", variant="primary")
                        t2_log = gr.Textbox(label="学習ログ", interactive=False, lines=8, elem_classes="status-box")

                t2_btn.click(
                    fn=step2_train,
                    inputs=[t2_image, t2_input_prompt, shared_output_prompt, t2_lr, t2_steps, t2_rank, t2_early_stop, gr.State(device)],
                    outputs=[t2_log],
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
                        t3_status = gr.Textbox(label="ステータス", interactive=False, lines=2, elem_classes="status-box")
                    with gr.Column(scale=2):
                        t3_gallery = gr.Gallery(label="生成フレーム", columns=5, height="auto")

                t3_btn.click(
                    fn=step3_generate,
                    inputs=[t3_output_prompt, t3_negative_prompt, t3_num_frames, t3_guidance, t3_attn_word, gr.State(device)],
                    outputs=[t3_gallery, t3_status],
                )

        # Tab 1 ボタン → Tab 2/3 コンポーネントを引き継ぎ（Tabs ブロック外に登録しUnboundLocalErrorを回避）
        t1_btn.click(
            fn=step1_generate_prompt,
            inputs=[t1_image, gr.State(device)],
            outputs=[t1_input_prompt, shared_output_prompt, t2_image, t2_input_prompt, t1_output_prompt, t3_output_prompt, t1_status],
        )

        # Step 1 プロンプト編集時に即座に Step 2/3 に反映
        t1_input_prompt.change(
            fn=lambda x: x,
            inputs=[t1_input_prompt],
            outputs=[t2_input_prompt],
        )
        t1_output_prompt.change(
            fn=lambda x: (x, x),
            inputs=[t1_output_prompt],
            outputs=[shared_output_prompt, t3_output_prompt],
        )

        t1_image.change(
            fn=lambda x: x,
            inputs=[t1_image],
            outputs=[t2_image],
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
