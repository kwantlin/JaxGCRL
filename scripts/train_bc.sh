#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

# Hardcoded checkpoint path for training
saved_ckpt_path="/home/kwantlin/JaxGCRL/runs/run_ant-main-meanfield_s_1/ckpt/step_11427840.pkl"

env=ant

for seed in 1 ; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python training_bc.py \
    --project_name test --group_name first_run --exp_name ${env}-bc-meanfield --num_evals 50 \
    --seed ${seed} --num_timesteps 10000000 --batch_size 256 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier 1 --saved_ckpt_path ${saved_ckpt_path} --log_wandb --disable_entropy_actor --mean_field
  done

echo "All runs have finished."
