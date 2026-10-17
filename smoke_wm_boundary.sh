#!/usr/bin/env bash
# ============================================================
# 临时冒烟测试: 跨 world-model off -> on 边界
#
# 目的:
#   用生产训练代码跑 2 个短 epoch, 专门跨一次 start_epoch 边界,
#   验证 DDP (find_unused_parameters=False) 在 "epoch 0 不用 world model
#   -> epoch 1 开始用 world model" 的切换下不会崩。
#
# 与正式训练的唯一区别 (全部通过 CLI 覆盖, 不改 yaml / 不改代码):
#   max_epochs=2          (正式 30)
#   start_epoch=1         (正式 15)
#   num_gpus=2            (本机只有 2 张卡)
#   limit_train_batches   (小比例, 缩短单 epoch)
#   limit_val_batches     (小比例)
#
# 通过后:
#   - 无需恢复任何文件 (yaml 里仍是 start_epoch=15)
#   - 直接 bash train_curriculum.sh 正式训练
# ============================================================

set -uo pipefail

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/Datasets/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/DrivoR/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/DrivoR"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250505/240906016/jjy/Datasets/navsim/dataset"
export HYDRA_FULL_ERROR=1

EXPERIMENT=SMOKE_wm_boundary_e1_v3
AGENT=drivoR
SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py"

echo "[smoke] experiment : ${EXPERIMENT}"
echo "[smoke] 验证目标   : epoch0 (world off) -> epoch1 (world on) 的 DDP 切换"

python "${SCRIPT}" \
    agent=${AGENT} \
    experiment_name=${EXPERIMENT} \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=2 \
    trainer.params.limit_train_batches=0.01 \
    trainer.params.limit_val_batches=0.01 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=8 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=2 \
    agent.progress_bar=false \
    agent.config.world_model.enabled=true \
    agent.config.world_model.start_epoch=1 \
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
