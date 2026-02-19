#!/usr/bin/env python3
"""
prompts3.json と images_out_old2 を入力として，
各サンプルについて横軸=フレーム, 縦軸=モデルのGIFビューワーHTMLを生成する。
フレームは viewer_frames/ ディレクトリに保存して相対パス参照（軽量版）。

Usage:
    python make_viewer.py <prompts_json> <gif_dir> [--output viewer.html] [--models m1 m2 ...]

Example:
    python make_viewer.py /work/DDIPM/kfukushima/wsim/prompts3.json \
                          /work/DDIPM/kfukushima/wsim/images_out_old2 \
                          --output /work/DDIPM/kfukushima/wsim/viewer.html
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageSequence


def extract_gif_frames(gif_path: Path, out_dir: Path, stem: str, model: str) -> list[str]:
    """GIFの各フレームをPNGファイルに保存し、相対パスのリストを返す（1フレーム飛ばし）"""
    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with Image.open(gif_path) as img:
        all_frames = list(ImageSequence.Iterator(img))
        for i, frame in enumerate(all_frames[::2]):  # 1つ飛ばし
            orig_idx = i * 2
            fname = f"{stem}_f{orig_idx:02d}.png"
            fpath = model_dir / fname
            if not fpath.exists():
                frame.convert("RGB").save(fpath, format="PNG")
            paths.append(f"viewer_frames/{model}/{fname}")
    return paths


def detect_models(gif_dir: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    return sorted(
        d.name for d in gif_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("output")
        and not d.name.startswith("_") and not d.name.startswith("proposed_old")
        and (d / "image_004.gif").exists()
    )


def copy_original(image_path: str, out_dir: Path, stem: str) -> str:
    """元画像をviewer_frames/original/にコピーして相対パスを返す"""
    orig_dir = out_dir / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)
    src = Path(image_path)
    dest = orig_dir / f"{stem}.png"
    if not dest.exists():
        Image.open(src).convert("RGB").save(dest, format="PNG")
    return f"viewer_frames/original/{stem}.png"


def build_html(samples: list[dict], gif_dir: Path, models: list[str], frames_dir: Path) -> str:
    # サンプルデータを構築（パス参照）
    js_data = []
    for i, sample in enumerate(samples):
        stem = Path(sample["image_path"]).stem
        orig_path = copy_original(sample["image_path"], frames_dir, stem)
        entry = {
            "idx": i,
            "stem": stem,
            "original_path": orig_path,
            "output_prompt": sample.get("output_prompt", ""),
            "models": {}
        }
        for model in models:
            gif_path = gif_dir / model / f"{stem}.gif"
            if gif_path.exists():
                frame_paths = extract_gif_frames(gif_path, frames_dir, stem, model)
                entry["models"][model] = frame_paths
            else:
                entry["models"][model] = []
        js_data.append(entry)
        print(f"  [{i+1}/{len(samples)}] {stem}", flush=True)

    data_json = json.dumps(js_data, ensure_ascii=False)
    models_json = json.dumps(models)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weathering GIF Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; }}

  header {{
    background: linear-gradient(135deg, #16213e, #0f3460);
    padding: 18px 28px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    position: sticky; top: 0; z-index: 100;
  }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; color: #e94560; }}

  header span {{ color: #aaa; font-size: 0.9rem; }}

  .controls {{
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    padding: 12px 28px;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
  }}
  .controls label {{ font-size: 0.85rem; color: #aaa; }}
  select, input {{ background: #0f3460; color: #e0e0e0; border: 1px solid #e94560; border-radius: 4px; padding: 4px 8px; }}
  input[type=range] {{ border: none; }}
  .btn {{
    background: #e94560; color: white; border: none; border-radius: 6px;
    padding: 6px 14px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
    transition: background 0.2s;
  }}
  .btn:hover {{ background: #c73652; }}
  .btn.secondary {{ background: #0f3460; }}
  .btn.secondary:hover {{ background: #1a4a8a; }}

  #sample-nav {{
    display: flex; align-items: center; gap: 6px;
    padding: 8px 28px; background: #1a1a2e;
    border-bottom: 1px solid #0f3460;
    overflow-x: auto;
  }}
  .nav-btn {{
    background: #16213e; border: 1px solid #0f3460; color: #e0e0e0;
    border-radius: 4px; padding: 4px 10px; cursor: pointer; font-size: 0.8rem;
    white-space: nowrap; transition: all 0.15s; flex-shrink: 0;
  }}
  .nav-btn.active {{ background: #e94560; border-color: #e94560; color: white; }}
  .nav-btn:hover:not(.active) {{ border-color: #e94560; color: #e94560; }}

  #main {{ padding: 16px 28px; overflow-x: auto; }}

  .sample-info {{
    margin-bottom: 12px; padding: 10px 14px;
    background: #16213e; border-left: 4px solid #e94560; border-radius: 0 8px 8px 0;
  }}
  .sample-info .label {{ font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }}
  .input-p {{ color: #8ecae6; font-size: 0.92rem; }}
  .output-p {{ color: #f9c74f; font-size: 0.92rem; }}

  table {{ border-collapse: collapse; min-width: max-content; }}
  th {{
    background: #0f3460; color: #e94560; font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; padding: 7px 8px;
    border-bottom: 2px solid #e94560; white-space: nowrap;
  }}
  th.model-col {{
    position: sticky; left: 0; z-index: 10; background: #0f3460;
    border-right: 2px solid #e94560; min-width: 110px;
  }}
  td {{ padding: 5px; border: 1px solid #0f3460; vertical-align: middle; }}
  td.model-name {{
    font-size: 0.85rem; font-weight: 700; color: #8ecae6;
    position: sticky; left: 0; z-index: 5; background: #16213e;
    border-right: 2px solid #e94560; padding: 6px 12px; white-space: nowrap;
  }}
  td.frame-cell {{ text-align: center; }}
  td.frame-cell img {{
    display: block; border-radius: 4px; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  td.frame-cell img:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 10px rgba(233,69,96,0.6);
  }}
  td.no-gif {{ color: #555; font-size: 0.8rem; text-align: center; }}
  tr:hover td {{ background: rgba(14,52,96,0.25); }}
  tr:hover td.model-name {{ background: #1e3e70; }}

  /* Lightbox */
  #lightbox {{
    display: none; position: fixed; inset: 0; z-index: 999;
    background: rgba(0,0,0,0.92); justify-content: center; align-items: center;
    flex-direction: column; gap: 12px;
  }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 90vw; max-height: 85vh; border-radius: 8px; }}
  #lightbox .lb-cap {{ color: #aaa; font-size: 0.85rem; text-align: center; max-width: 80vw; }}
  #lightbox .lb-close {{
    position: absolute; top: 16px; right: 22px;
    color: #e94560; font-size: 2rem; cursor: pointer; font-weight: 700;
  }}
</style>
</head>
<body>

<header>
  <h1>Weathering GIF Viewer</h1>
  <span id="hdr-info"></span>
</header>

<div class="controls">
  <label>画像サイズ:</label>
  <input type="range" id="img-size" min="60" max="300" value="192" step="10">
  <span id="size-label" style="font-size:0.8rem;color:#aaa">192px</span>

  <label style="margin-left:16px">絞り込み:</label>
  <input type="text" id="filter-input" placeholder="stem名" style="width:160px">

  <button class="btn secondary" onclick="prevSample()">◀ 前へ</button>
  <button class="btn secondary" onclick="nextSample()">次へ ▶</button>
</div>

<div id="sample-nav"></div>

<div id="main">
  <div class="sample-info">
    <div class="label">Output Prompt</div>
    <div class="output-p" id="info-output"></div>
  </div>
  <table id="grid">
    <thead id="grid-head"></thead>
    <tbody id="grid-body"></tbody>
  </table>
</div>

<div id="lightbox">
  <span class="lb-close" onclick="closeLightbox()">✕</span>
  <img id="lb-img" src="" alt="">
  <div class="lb-cap" id="lb-cap"></div>
</div>

<script>
const DATA = {data_json};
const MODELS = {models_json};
let currentIdx = 0;
let imgSize = 192;

function buildNav() {{
  const nav = document.getElementById('sample-nav');
  nav.innerHTML = '';
  DATA.forEach((s, i) => {{
    const btn = document.createElement('button');
    btn.className = 'nav-btn' + (i === currentIdx ? ' active' : '');
    btn.textContent = s.stem;
    btn.onclick = () => showSample(i);
    nav.appendChild(btn);
  }});
}}

function renderGrid(idx) {{
  const s = DATA[idx];
  document.getElementById('info-output').textContent = s.output_prompt;
  document.getElementById('hdr-info').textContent = `[${{idx+1}}/${{DATA.length}}] ${{s.stem}}`;

  let maxFrames = 0;
  MODELS.forEach(m => {{ maxFrames = Math.max(maxFrames, (s.models[m] || []).length); }});

  // Header
  const head = document.getElementById('grid-head');
  head.innerHTML = '';
  const tr = document.createElement('tr');
  const thm = document.createElement('th');
  thm.className = 'model-col';
  thm.textContent = 'Model';
  tr.appendChild(thm);
  for (let f = 0; f < maxFrames; f++) {{
    const th = document.createElement('th');
    th.textContent = `Frame ${{f+1}}`;
    tr.appendChild(th);
  }}
  head.appendChild(tr);

  // Body
  const body = document.getElementById('grid-body');
  body.innerHTML = '';

  // --- Original 行 ---
  const origRow = document.createElement('tr');
  const origName = document.createElement('td');
  origName.className = 'model-name';
  origName.textContent = 'Original';
  origName.style.color = '#f9c74f';
  origRow.appendChild(origName);
  const origTd = document.createElement('td');
  origTd.className = 'frame-cell';
  if (s.original_path) {{
    const img = document.createElement('img');
    img.src = s.original_path;
    img.width = imgSize; img.height = imgSize;
    img.loading = 'lazy';
    img.alt = 'Original';
    img.onclick = () => openLightbox(s.original_path, `Original / ${{s.stem}}`);
    origTd.appendChild(img);
  }}
  origRow.appendChild(origTd);
  // 空セルで残りを埋める
  for (let f = 1; f < maxFrames; f++) {{
    const td = document.createElement('td');
    origRow.appendChild(td);
  }}
  body.appendChild(origRow);

  // --- モデル行 ---
  MODELS.forEach(model => {{
    const frames = s.models[model] || [];
    const row = document.createElement('tr');

    const tdName = document.createElement('td');
    tdName.className = 'model-name';
    tdName.textContent = model;
    row.appendChild(tdName);

    for (let f = 0; f < maxFrames; f++) {{
      const td = document.createElement('td');
      td.className = 'frame-cell';
      if (f < frames.length) {{
        const img = document.createElement('img');
        img.src = frames[f];
        img.width = imgSize;
        img.height = imgSize;
        img.loading = 'lazy';
        img.alt = `${{model}} frame ${{f+1}}`;
        img.onclick = () => openLightbox(frames[f], `${{model}} / Frame ${{f+1}} / ${{s.stem}}`);
        td.appendChild(img);
      }} else {{
        td.className = 'no-gif';
        td.textContent = '—';
      }}
      row.appendChild(td);
    }}
    body.appendChild(row);
  }});
}}

function showSample(idx) {{
  currentIdx = idx;
  document.querySelectorAll('.nav-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
  renderGrid(idx);
}}
function prevSample() {{ showSample(Math.max(0, currentIdx - 1)); }}
function nextSample() {{ showSample(Math.min(DATA.length - 1, currentIdx + 1)); }}

function openLightbox(src, cap) {{
  document.getElementById('lb-img').src = src;
  document.getElementById('lb-cap').textContent = cap;
  document.getElementById('lightbox').classList.add('open');
}}
function closeLightbox() {{ document.getElementById('lightbox').classList.remove('open'); }}
document.getElementById('lightbox').addEventListener('click', e => {{
  if (e.target === document.getElementById('lightbox')) closeLightbox();
}});

document.getElementById('img-size').addEventListener('input', function() {{
  imgSize = parseInt(this.value);
  document.getElementById('size-label').textContent = imgSize + 'px';
  document.querySelectorAll('#grid-body img').forEach(img => {{
    img.width = imgSize; img.height = imgSize;
  }});
}});

document.getElementById('filter-input').addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  document.querySelectorAll('.nav-btn').forEach((b, i) => {{
    b.style.display = DATA[i].stem.toLowerCase().includes(q) ? '' : 'none';
  }});
}});

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') nextSample();
  if (e.key === 'ArrowLeft') prevSample();
  if (e.key === 'Escape') closeLightbox();
}});

buildNav();
renderGrid(0);
</script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate GIF viewer HTML (lightweight)")
    parser.add_argument("prompts_json", help="prompts3.json のパス")
    parser.add_argument("gif_dir", help="images_out_old2 ディレクトリ")
    parser.add_argument("--output", default=None, help="出力HTMLパス")
    parser.add_argument("--models", nargs="+", default=None, help="表示するモデル名")
    args = parser.parse_args()

    json_path = Path(args.prompts_json)
    gif_dir = Path(args.gif_dir)

    with open(json_path, encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict):
        samples = [samples]


    models = detect_models(gif_dir, args.models)
    print(f"Models: {models}")
    print(f"Samples: {len(samples)}")

    output_path = Path(args.output) if args.output else gif_dir.parent / "viewer.html"
    # フレーム保存先: HTML と同じディレクトリの viewer_frames/
    frames_dir = output_path.parent / "viewer_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting frames...")
    html = build_html(samples, gif_dir, models, frames_dir)

    output_path.write_text(html, encoding="utf-8")
    print(f"\nDone! Saved: {output_path}")
    print(f"Frames dir: {frames_dir}")
    print(f"Open with: xdg-open '{output_path}'")


if __name__ == "__main__":
    main()
