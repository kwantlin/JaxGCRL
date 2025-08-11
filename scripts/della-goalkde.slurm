#!/bin/bash

#SBATCH -J della-train-ant-fullobs-goalkde-meanfield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH --partition=pli 
#SBATCH --account=buildstuff
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

env=ant_fullobs

for seed in 1 ; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python training_goalkde.py \
    --project_name test --group_name first_run --exp_name ${env}-goalkde-meanfield --num_evals 50 \
    --seed ${seed} --num_timesteps 12000000000 --batch_size 512 --num_envs 2048 \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier 1 --log_wandb --var_post meanfield
  done
  
echo "All runs have finished."