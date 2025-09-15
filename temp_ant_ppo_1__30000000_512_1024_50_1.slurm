#!/bin/bash

#SBATCH --job-name=ant_ppo_1__30000000_512_1024_50_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 30:00:00
#SBATCH --partition=pli 
#SBATCH --account=buildstuff
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu
#SBATCH --output=logs/della_ant_ppo_1__30000000_512_1024_50_1.out
#SBATCH --error=logs/della_ant_ppo_1__30000000_512_1024_50_1.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl


python training_ppo.py     --project_name test --group_name first_run --exp_name ant-ppo-numenvs1024-numtimesteps30000000-batchsize512-1-della-maxent-gaussianmlp- --num_evals 50     --seed 1 --num_timesteps 30000000 --batch_size 512 --num_envs 1024     --discounting 0.99 --action_repeat 1 --env_name ant --use_dense_reward     --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 --log_wandb
  
