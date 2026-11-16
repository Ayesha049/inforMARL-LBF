#!/bin/bash
# Train one InforMARL policy on LBF. PPO / network hyper-parameters are exactly those of the
# Phase 0a Target reproduction (inforMARL-rep/scripts/repro/train_one.sh); only the env differs.
#
# usage: bash scripts/train_one.sh <kappa> <num_agents> <seed> <num_env_steps> [gpu_id] [extra args...]
set -euo pipefail
K=$1; N=$2; SEED=$3; STEPS=$4; GPU=${5:-0}; shift $(( $# < 5 ? $# : 5 ))
cd "$(dirname "$0")/.."
mkdir -p logs
EXP="lbf_k${K}_n${N}_s${SEED}"
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONNOUSERSITE=1
python -u onpolicy/scripts/train_lbf.py --use_valuenorm --use_popart \
  --algorithm_name "rmappo" \
  --seed ${SEED} \
  --experiment_name "${EXP}" \
  --kappa ${K} --num_agents ${N} \
  --n_training_threads 1 --n_rollout_threads 128 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps ${STEPS} \
  --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
  --user_name "marl" \
  --use_cent_obs "False" \
  --graph_feat_type "relative" \
  --auto_mini_batch_size --target_mini_batch_size 128 \
  "$@" \
  > logs/train_${EXP}.log 2>&1 || true

LOG=logs/train_${EXP}.log
if grep -q "Traceback" "$LOG" || ! grep "Percentage complete" "$LOG" | tail -1 | grep -qE "complete (99|100)"; then
  echo "TRAINING FAILED or incomplete: ${EXP}" >&2; exit 1
fi
