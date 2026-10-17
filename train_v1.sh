#!/usr/bin/env bash
# DrivoR NAVSIM-v1 训练启动脚本 (长轨迹 25 epochs)
#
# 用法:
#   bash train_drivoR.sh                          # 使用下方默认参数
#   bash train_drivoR.sh agent.lr_args.base_lr=0.0001   # 追加/覆盖任意 hydra 参数
#
# 说明:
#   - 环境变量用 ${VAR:-默认值},因此外部已 export 的值优先,不会被覆盖
#   - 使用 set -u 而非 set -e: 训练异常退出时保留现场,便于查看日志

set -uo pipefail

########################################
# 环境变量
########################################
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/Datasets/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/DrivoR/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/DrivoR"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/Datasets/navsim/dataset"

export HYDRA_FULL_ERROR=1

########################################
# 训练参数
########################################
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs_world_model_2
AGENT=drivoR

SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py"

echo "[train] experiment : ${EXPERIMENT}"
echo "[train] agent      : ${AGENT}"
echo "[train] exp root   : ${NAVSIM_EXP_ROOT}"
echo "[train] script     : ${SCRIPT}"

########################################
# 启动训练
########################################
python "${SCRIPT}" \
    agent=${AGENT} \
    experiment_name=${EXPERIMENT} \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2 \
    "$@"
