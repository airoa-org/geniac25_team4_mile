## Robot MILE Datasets (HSR / OXE Fractal)

### 概要

- HSR と OXE Fractal (IPEC-COMMUNITY/fractal20220817_data_lerobot) の両データセットに対応
- LeRobotライクな設計（メタデータ駆動、モダリティ統一: video/state/action/language）
- AV1コーデック問題に対するフォールバック、動画キャッシュ、高速データロード

### 実装ファイル

```
mile/data/hsr_dataset.py           # HSR用 Dataset / DataModule
mile/data/oxe_fractal_dataset.py   # OXE Fractal 用 Dataset / DataModule
train_hsr.py                       # 共通トレーニングスクリプト（--dataset_type で切替）
```

## セットアップ

コンテナビルド（必要なら）:

```
bash sh/build.sh
```

コンテナ起動（fractal を /opt/processed/fractal20220817_data_lerobot にバインド）:

```
bash sh/run.sh
```

コンテナ内で依存が不足する場合は以下を実行してください。

```
python3 -m pip install av pandas pyarrow
```

## OXE Fractal での学習

### データ構造

- ルート: `/opt/processed/fractal20220817_data_lerobot`
- Hugging Face スタイルの `snapshots/<hash>/` を自動検出
- 例:
```
/opt/processed/fractal20220817_data_lerobot/
  snapshots/<hash>/
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.image/episode_XXXXXX.mp4
    meta/tasks.jsonl
```

### 学習コマンド

```
python3 train_hsr.py \
  --dataset_type oxe_fractal \
  --data_root /opt/processed/fractal20220817_data_lerobot \
  --camera images.image \
  --sequence_length 8 \
  --batch_size 4 \
  --num_workers 4 \
  --img_resize 320 240 \
  --use_wandb
```

ポイント:
- `--dataset_type oxe_fractal` を指定
- カメラは `--camera images.image`（内部ディレクトリ `observation.images.image` を使用）

## HSR での学習（参考）

```
python3 train_hsr.py \
  --dataset_type hsr \
  --data_root /path/to/hsr_data \
  --camera head_rgbd_sensor \
  --sequence_length 8 \
  --batch_size 4 \
  --num_workers 4 \
  --use_wandb
```

## よくある問題

- Parquet 読み込みでエラー: `pandas` / `pyarrow` が必要です。上記の pip インストールを実行してください。
- AV1 動画が読めない: デフォルトで PyAV バックエンド（FFmpeg）を使用。必要に応じて H264 へ変換してください。
- データパスが検出されない: `snapshots/<hash>/` 直下に `data/` と `videos/` があるか確認してください。

## 設計メモ

- `OXEFractalDataset` は `snapshots/<hash>/` を自動検出し、`data/` と `videos/` を探索します
- 動画キーは `video.images.image`（カメラディレクトリは `observation.images.image`）
- 言語モダリティは `meta/tasks.jsonl` の `task_index` からタスク文字列を復元


