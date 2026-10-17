#!/usr/bin/env bash
# DrivoR NAVSIM-v1 Curriculum Training (15 epochs DrivoR-only + 15 epochs DrivoR+WorldModel)
#
# 实验目的:
#   验证 "延迟到第 15 epoch 再加入 world-model supervision, 能否避免
#   early-stage negative interference"。
#
# 关键性质:
#   - 单一训练进程, 单一 optimizer / scheduler / scaler
#   - 连续 30 epoch LR schedule (不因开启 world model 而重置)
#   - epoch  0~14: 只算 L_DrivoR, world-model branch 整体跳过
#   - epoch 15~29: L_DrivoR + 0.03 * L_world
#   - 每个 epoch 的 checkpoint 只保留最近 10 轮 (KEEP_LAST_N=10)
#
# 用法:
#   bash train_curriculum.sh                              # 使用下方默认参数
#   bash train_curriculum.sh agent.lr_args.base_lr=0.0001 # 追加/覆盖任意 hydra 参数
#
# 说明:
#   - 环境变量用 ${VAR:-默认值}, 因此外部已 export 的值优先, 不会被覆盖
#   - 使用 set -u 而非 set -e: 训练异常退出时保留现场, 便于查看日志

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
EXPERIMENT=training_drivoR_Nav1_traj_long_30epochs_wm_curriculum_e15_1
AGENT=drivoR

SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py"

# curriculum 切换点 (与 drivoR.yaml 的 world_model.start_epoch 保持一致, 便于阅读)
WM_START_EPOCH=15
MAX_EPOCHS=30

echo "[curriculum] experiment   : ${EXPERIMENT}"
echo "[curriculum] agent        : ${AGENT}"
echo "[curriculum] wm start ep  : ${WM_START_EPOCH}"
echo "[curriculum] max epochs   : ${MAX_EPOCHS}"
echo "[curriculum] exp root     : ${NAVSIM_EXP_ROOT}"
echo "[curriculum] script       : ${SCRIPT}"

########################################
# 启动训练 (从零训练, 不传 warm_start_path)
########################################
python "${SCRIPT}" \
    agent=${AGENT} \
    experiment_name=${EXPERIMENT} \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=${MAX_EPOCHS} \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.world_model.enabled=true \
    agent.config.world_model.start_epoch=${WM_START_EPOCH} \
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
