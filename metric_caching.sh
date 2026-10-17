#!/usr/bin/env bash
# NAVSIM-v1 navtest split metric cache 生成脚本
#
# 用法:
#   bash metric_caching.sh                          # 使用下方默认参数
#   bash metric_caching.sh worker.threads_per_node=16   # 追加/覆盖任意 hydra 参数
#
# 说明:
#   - 评测 (eval_drivoR.sh) 依赖本脚本产出的缓存; 缓存缺失或为空的直接表现是
#     MetricCacheLoader 报 FileNotFoundError: <cache_path>/metadata
#   - 缓存目录布局: <cache_path>/<log_name>/<token>/metric_cache.pkl
#                   <cache_path>/metadata/*.csv  (csv 记录全部 pkl 路径, 评测靠它索引)
#   - 环境变量用 ${VAR:-默认值}, 因此外部已 export 的值优先, 不会被覆盖
#   - 使用 set -u 而非 set -e: 异常退出时保留现场, 便于查看日志

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
# 缓存参数
########################################
TRAIN_TEST_SPLIT=navtest
CACHE_PATH=${NAVSIM_EXP_ROOT}/metric_cache

SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py"

echo "[cache] split      : ${TRAIN_TEST_SPLIT}"
echo "[cache] cache path : ${CACHE_PATH}"
echo "[cache] exp root   : ${NAVSIM_EXP_ROOT}"
echo "[cache] script     : ${SCRIPT}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "[cache] ERROR: script not found: ${SCRIPT}" >&2
    exit 1
fi

########################################
# 生成缓存
########################################
python "${SCRIPT}" \
    train_test_split=${TRAIN_TEST_SPLIT} \
    cache.cache_path="${CACHE_PATH}" \
    "$@"
