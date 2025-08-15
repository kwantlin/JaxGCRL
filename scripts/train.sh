#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

method=crl
env=sudoku

for seed in 1 2 3 4 5 ; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python main.py "$method" \
    --wandb_project_name test --wandb_group first_run --exp_name ${env}_${seed} --num_evals 50 \
    --seed ${seed} --total_env_steps 800000000 --batch_size 256 --num_envs 2048 \
    --discounting 0.99 --action_repeat 1 --env ${env} \
    --episode_length 240 --unroll_length 240  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn bwd_infonce --energy_fn norm \
    --train_step_multiplier 4 --log_wandb --repetition_factor 2
  done

echo "All runs have finished."
