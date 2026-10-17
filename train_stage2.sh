#!/usr/bin/env bash
# DrivoR NAVSIM-v1 两阶段训练 - 第二阶段 (热启动 + world model 联合训练)
#
# 阶段说明:
#   stage-1: 无 world model, 从头训练主体             -> 已完成
#   stage-2: 用 stage-1 权重热启动, 打开 world model 联合训练 (主体全参数可训)
#
# 用法:
#   bash train_stage2.sh                                   # 使用下方默认参数
#   bash train_stage2.sh agent.lr_args.base_lr=0.0001      # 追加/覆盖任意 hydra 参数
#
# 前置条件:
#   WARM_START_CKPT 指向的 stage-1 checkpoint 必须存在。
#
# 说明:
#   - 环境变量用 ${VAR:-默认值},因此外部已 export 的值优先,不会被覆盖
#   - 使用 set -u 而非 set -e: 训练异常退出时保留现场,便于查看日志
#   - 注意: checkpoint 文件名含 "=", 必须用单引号包裹整个值,
#     否则 hydra 解析 override 时会报 OverrideParseException

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
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs_world_model_stage2
AGENT=drivoR

SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py"

# stage-1 (无 world model) 训练出的主体权重
WARM_START_CKPT="${NAVSIM_EXP_ROOT}/ke/training_drivoR_Nav1_traj_long_25epochs/09.25_03.43/lightning_logs/version_0/checkpoints/best-epoch=23-step=38472.ckpt"

echo "[stage2] experiment  : ${EXPERIMENT}"
echo "[stage2] agent       : ${AGENT}"
echo "[stage2] warm-start  : ${WARM_START_CKPT}"
echo "[stage2] exp root    : ${NAVSIM_EXP_ROOT}"
echo "[stage2] script      : ${SCRIPT}"

if [[ ! -f "${WARM_START_CKPT}" ]]; then
    echo "[stage2] ERROR: warm-start checkpoint not found: ${WARM_START_CKPT}" >&2
    exit 1
fi

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
    agent.warm_start_path="'${WARM_START_CKPT}'" \
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
