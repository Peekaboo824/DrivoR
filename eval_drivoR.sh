#!/usr/bin/env bash
# DrivoR NAVSIM-v1 评测脚本 (长轨迹 25 epochs, navtest split)
#
# 用法:
#   bash eval_drivoR.sh                                  # 使用下方默认参数
#   bash eval_drivoR.sh agent.config.proposal_num=128    # 追加/覆盖任意 hydra 参数
#
# 前置条件:
#   metric_cache 必须已生成, 否则评测会因命中不到 metric cache 而失败。
#   若尚未生成, 先执行:  bash metric_caching.sh
#
# 说明:
#   - 环境变量用 ${VAR:-默认值}, 因此外部已 export 的值优先, 不会被覆盖
#   - 使用 set -u 而非 set -e: 评测异常退出时保留现场, 便于查看日志

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

# run_pdm_score_multi_gpu.py 读取 SUBSCORE_PATH 作为 pkl 落盘根目录
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT

export HYDRA_FULL_ERROR=1

########################################
# 评测参数
########################################
EXPERIMENT=drivoR_nav1
AGENT=drivoR

# 训练 25 epochs 输出的 lightning checkpoint (内含 agent.* 前缀的 state_dict, 可直接加载)
# 注意: 文件名含 "=", 必须用单引号包裹整个值, 否则 hydra 解析 override 时报 OverrideParseException
CKPT="/vepfs-mlp2/c20250505/240906016/jjy/DrivoR/exp/ke/training_drivoR_Nav1_traj_long_30epochs_wm_curriculum_e15_1/10.14_16.31/lightning_logs/version_0/checkpoints/epoch-epoch=19.ckpt"

SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py"

echo "[eval] experiment : ${EXPERIMENT}"
echo "[eval] agent      : ${AGENT}"
echo "[eval] checkpoint : ${CKPT}"
echo "[eval] exp root   : ${NAVSIM_EXP_ROOT}"
echo "[eval] script     : ${SCRIPT}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[eval] ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

########################################
# 启动评测
########################################
python "${SCRIPT}" \
    train_test_split=navtest \
    agent=${AGENT} \
    agent.checkpoint_path="'${CKPT}'" \
    experiment_name=${EXPERIMENT} \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc=1 \
    agent.config.dac=1 \
    agent.config.ddc=0.0 \
    agent.config.ttc=5 \
    agent.config.ep=5 \
    agent.config.comfort=2 \
    agent.config.world_model.enabled=false \
    "$@"
