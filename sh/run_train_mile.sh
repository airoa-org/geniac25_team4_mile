#!/bin/bash

#SBATCH --partition part-group_25b505
#SBATCH --nodelist aic-gh2b-310052
#SBATCH --nodes 1
#SBATCH --time 12:00:00
#SBATCH --output /home/group_25b505/group_4/members/koen/geniac25_team4_mile/logs/%x-%j.out
#SBATCH --gpus 1

module load singularitypro
module load cuda

# Singularity の一時領域を /home/.../tmp に切り替え
export SINGULARITY_TMPDIR=/home/group_25b505/group_4/members/koen/tmp
export TMPDIR=$SINGULARITY_TMPDIR
mkdir -p $SINGULARITY_TMPDIR

# マウント先ディレクトリ
export TARGET_DIR=/home/group_25b505/group_4/members/koen/geniac25_team4_mile

# データセット用バインド先
DATASET_HOST=/home/group_25b505/group_4/datasets
DATASET_CONT=/opt/processed

# コンテナ内で直接 python を叩く
singularity exec --fakeroot --nv \
  --bind $DATASET_HOST:$DATASET_CONT \
  --bind $TARGET_DIR:/workspace \
  $TARGET_DIR/robot_mile.sif \
  python3 /workspace/train_hsr_v2.py \
    --data_root data/tmc_new \
    --use_wandb
