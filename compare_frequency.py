"""
2枚の画像を周波数帯域ごとに比較するプログラム
ラプラシアンピラミッドを使用して異なる周波数帯域を分解し、比較可能な形式で出力
カラー対応版・細かい周波数分解
"""

import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import japanize_matplotlib


def build_laplacian_pyramid_color(image: np.ndarray, levels: int = 6) -> list[np.ndarray]:
    """カラー画像用ラプラシアンピラミッドを構築"""
    pyramid = []
    current = image.astype(np.float32)
    
    for i in range(levels - 1):
        # ダウンサンプリング
        down = cv2.pyrDown(current)
        # アップサンプリングして差分を取る（高周波成分）
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        laplacian = current - up
        pyramid.append(laplacian)
        current = down
    
    # 最低周波数成分（残差）
    pyramid.append(current)
    
    return pyramid


def normalize_for_display_color(image: np.ndarray) -> np.ndarray:
    """カラー画像を表示用に正規化（ラプラシアン成分は中央を128として）"""
    # ラプラシアン成分は負の値を含むので、中央を128にシフト
    normalized = image + 128
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def compute_difference_color(pyramid1: list[np.ndarray], pyramid2: list[np.ndarray]) -> list[np.ndarray]:
    """各レベルの差分を計算（カラー）"""
    diff_pyramid = []
    for p1, p2 in zip(pyramid1, pyramid2):
        # サイズを合わせる
        if p1.shape != p2.shape:
            p2 = cv2.resize(p2, (p1.shape[1], p1.shape[0]))
        diff = np.abs(p1 - p2)
        diff_pyramid.append(diff)
    return diff_pyramid


def compare_frequency_bands(image1_path: str, image2_path: str, output_path: str, levels: int = 6):
    """2枚の画像を周波数帯域ごとに比較（カラー版）"""
    
    # 画像を読み込み
    img1 = np.array(Image.open(image1_path).convert("RGB"))
    img2 = np.array(Image.open(image2_path).convert("RGB"))
    
    # サイズを統一
    target_size = (512, 512)
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)
    
    # カラーでラプラシアンピラミッドを構築
    pyramid1 = build_laplacian_pyramid_color(img1, levels)
    pyramid2 = build_laplacian_pyramid_color(img2, levels)
    
    # 差分を計算
    diff_pyramid = compute_difference_color(pyramid1, pyramid2)
    
    # 周波数帯域の名前を生成
    band_names = []
    for i in range(levels - 1):
        if i == 0:
            band_names.append("最高周波数（細部）")
        elif i == levels - 2:
            band_names.append("中間周波数")
        else:
            band_names.append(f"高周波 {i+1}")
    band_names.append("最低周波数（全体構造）")
    
    # 可視化
    fig, axes = plt.subplots(levels, 4, figsize=(16, 3 * levels))
    
    for i in range(levels):
        # サイズを表示用に調整
        display_size = (256, 256)
        
        # 画像1のバンド（カラー）
        if i < levels - 1:
            band1 = cv2.resize(normalize_for_display_color(pyramid1[i]), display_size)
        else:
            band1 = cv2.resize(np.clip(pyramid1[i], 0, 255).astype(np.uint8), display_size)
        axes[i, 0].imshow(band1)
        axes[i, 0].set_title(f"画像1: {band_names[i]}")
        axes[i, 0].axis('off')
        
        # 画像2のバンド（カラー）
        if i < levels - 1:
            band2 = cv2.resize(normalize_for_display_color(pyramid2[i]), display_size)
        else:
            band2 = cv2.resize(np.clip(pyramid2[i], 0, 255).astype(np.uint8), display_size)
        axes[i, 1].imshow(band2)
        axes[i, 1].set_title(f"画像2: {band_names[i]}")
        axes[i, 1].axis('off')
        
        # 差分（カラーマップで強調）
        diff = cv2.resize(diff_pyramid[i], display_size)
        # RGBの差分を合成して強度マップに
        diff_intensity = np.mean(diff, axis=2)
        diff_normalized = (diff_intensity / (diff_intensity.max() + 1e-8) * 255).astype(np.uint8)
        axes[i, 2].imshow(diff_normalized, cmap='hot')
        axes[i, 2].set_title(f"差分強度: {band_names[i]}")
        axes[i, 2].axis('off')
        
        # RGB各チャンネルの差分統計
        r_diff = np.mean(diff_pyramid[i][:, :, 0])
        g_diff = np.mean(diff_pyramid[i][:, :, 1])
        b_diff = np.mean(diff_pyramid[i][:, :, 2])
        total_diff = np.mean(diff_pyramid[i])
        
        stats_text = f"R差分: {r_diff:.2f}\nG差分: {g_diff:.2f}\nB差分: {b_diff:.2f}\n合計: {total_diff:.2f}"
        axes[i, 3].text(0.5, 0.5, stats_text,
                        ha='center', va='center', fontsize=10, transform=axes[i, 3].transAxes,
                        family='monospace')
        axes[i, 3].set_title(f"統計: {band_names[i]}")
        axes[i, 3].axis('off')
    
    fig.suptitle("周波数帯域別比較（カラー）", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存先: {output_path}")
    
    # 追加: 元画像と全体比較を別ファイルに保存
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    axes2[0].imshow(img1)
    axes2[0].set_title("画像1（入力）")
    axes2[0].axis('off')
    
    axes2[1].imshow(img2)
    axes2[1].set_title("画像2（出力）")
    axes2[1].axis('off')
    
    # 全体差分（カラー）
    diff_total = cv2.absdiff(img1, img2)
    axes2[2].imshow(diff_total)
    axes2[2].set_title("全体差分（カラー）")
    axes2[2].axis('off')
    
    # 差分強度マップ
    diff_intensity = np.mean(diff_total.astype(np.float32), axis=2)
    axes2[3].imshow(diff_intensity, cmap='hot')
    axes2[3].set_title("差分強度マップ")
    axes2[3].axis('off')
    
    overview_path = output_path.replace('.png', '_overview.png')
    plt.tight_layout()
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"概要図: {overview_path}")
    
    # 各チャンネル別の周波数比較も出力
    channel_names = ['赤(R)', '緑(G)', '青(B)']
    fig3, axes3 = plt.subplots(3, levels, figsize=(3 * levels, 9))
    
    for ch, ch_name in enumerate(channel_names):
        for lvl in range(levels):
            diff = diff_pyramid[lvl][:, :, ch]
            diff_resized = cv2.resize(diff, (128, 128))
            diff_normalized = (diff_resized / (diff_resized.max() + 1e-8) * 255).astype(np.uint8)
            axes3[ch, lvl].imshow(diff_normalized, cmap='hot')
            if ch == 0:
                axes3[ch, lvl].set_title(f"Level {lvl+1}", fontsize=10)
            if lvl == 0:
                axes3[ch, lvl].set_ylabel(ch_name, fontsize=12)
            axes3[ch, lvl].axis('off')
    
    fig3.suptitle("チャンネル別 × 周波数帯域別 差分", fontsize=14)
    channel_path = output_path.replace('.png', '_channels.png')
    plt.tight_layout()
    plt.savefig(channel_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"チャンネル別: {channel_path}")


def main():
    parser = argparse.ArgumentParser(description="2枚の画像を周波数帯域ごとに比較（カラー対応）")
    parser.add_argument("--image1", type=str, required=True, help="1枚目の画像パス")
    parser.add_argument("--image2", type=str, required=True, help="2枚目の画像パス")
    parser.add_argument("--output", type=str, default="frequency_comparison.png", help="出力ファイルパス")
    parser.add_argument("--levels", type=int, default=6, help="ピラミッドのレベル数（デフォルト: 6）")
    
    args = parser.parse_args()
    
    compare_frequency_bands(args.image1, args.image2, args.output, args.levels)


if __name__ == "__main__":
    main()
