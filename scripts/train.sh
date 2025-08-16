#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate jaxgcrlsudoku

method=crl
env=sudoku

for seed in 1 ; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python main.py "$method" \
    --wandb_project_name sudoku-bmm --wandb_group first_run --exp_name ${env}_${seed} --num_evals 1000 \
    --seed ${seed} --total_env_steps 200000000000000 --batch_size 512 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env ${env} \
    --episode_length 240 --unroll_length 240  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn bwd_infonce --energy_fn norm \
    --train_step_multiplier 16 --repetition_factor 8 --log_wandb --alpha_lr 1e-3 --policy_lr 1e-4 --critic_lr 1e-4
  done

echo "All runs have finished."
