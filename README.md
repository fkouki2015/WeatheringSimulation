# WeatheringSimulation： シーン記述にもとづく局所制御可能な潜在拡散モデルを用いた画像の経年変化シミュレーション

単一画像のみを入力とし，ユーザー補助無しで経年変化を時系列順にシミュレーションします．

## クイックスタートガイド

### 前提条件

  - CUDAをサポートするNVIDIA GPU
  - Anaconda/Minicondaがインストール済みであること
  - 20GB以上のGPUメモリを推奨


### 1. 環境設定

conda環境のセットアップ

```bash
conda env create -f environment.yml
conda activate weathering
```

### 2. シミュレーションの実行

`main.py`を通じてシミュレーションを実行します．単一の画像を処理するか，フォルダ内のすべての画像を処理するかを選択できます．

#### 単一画像の処理

特定の1枚の画像を経年変化させる場合：

```bash
python main.py --input_image "path/to/image.jpg" --output_folder "outputs"
```

#### フォルダの一括処理

フォルダ内のすべての画像を一括で処理する場合：

```bash
python main.py --input_folder "path/to/image_folder" --output_folder "outputs"
```

### 3. オプション機能

生成結果をより細かく制御するためのオプション機能


#### 手動プロンプト指定 (`--train_prompt`, `--inference_prompt`)

デフォルトでは視覚言語モデルが自動で画像の説明と推論プロンプトを生成しますが，手動で指定することも可能です．

- `--train_prompt`: 画像を説明するキャプション
- `--inference_prompt`: 経年変化後の画像を説明するキャプション

```bash
python main.py \
  --input_image "car.jpg" \
  --train_prompt "a photo of a sports car" \
  --inference_prompt "a photo of a rusted sports car"
```


#### アテンション強調ワード (`--attn_word`)

プロンプト中の特定のキーワードのアテンションを強調し，その効果を強めることができます．
手動プロンプトとの併用を推奨します．

```bash
python main.py \
  --input_image "car.jpg" \
  --train_prompt "a photo of a sports car" \
  --inference_prompt "a photo of a rusted sports car"\
  --attn_word "rusted"
```

#### モード指定 (`--mode`)

デフォルトでは経年変化をシミュレーションしますが，劣化した画像からの復元をシミュレーションすることも可能です．

- `--mode`: "age"または"restore"
- "age"モード（デフォルト）： 経年変化シミュレーション
- "restore"モード： 劣化した画像からの復元シミュレーション

```bash
python main.py \
  --input_image "rusted_car.jpg" \
  --mode "restore"
```