# WeatheringSimulation： シーン記述にもとづく局所制御可能な潜在拡散モデルを用いた画像の経年変化シミュレーション

単一画像のみを入力とし，ユーザー補助無しで経年変化を時系列順にシミュレーションする．

## クイックスタートガイド

### 前提条件

  - CUDAをサポートするNVIDIA GPU
  - Anaconda/Minicondaがインストール済みであること
  - 20GB以上のGPUメモリを推奨


### 1\. 環境設定

conda環境のセットアップ

```bash
# flux環境を作成して有効化
conda env create -f environment.yml
conda activate weathering
```
