#!/bin/bash

#SBATCH --job-name=reacher_ppo_1_newpponet_2000000000_1024_256_100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --partition=pli 
#SBATCH --account=buildstuff
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu
#SBATCH --output=logs/della_reacher_ppo_1_newpponet_2000000000_1024_256_100.out
#SBATCH --error=logs/della_reacher_ppo_1_newpponet_2000000000_1024_256_100.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl


python training_ppo.py     --project_name test --group_name first_run --exp_name reacher-ppo-numenvs256-numtimesteps2000000000-batchsize1024-della-maxent-gaussianmlp-newpponet --num_evals 100     --seed 1 --num_timesteps 2000000000 --batch_size 1024 --num_envs 256     --discounting 0.99 --action_repeat 1 --env_name reacher --use_dense_reward     --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 --log_wandb
  
