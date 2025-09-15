#!/bin/bash

#SBATCH --job-name=reacher_ppo_1__20000000_1024_256_50_1
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
#SBATCH --output=logs/della_reacher_ppo_1__20000000_1024_256_50_1.out
#SBATCH --error=logs/della_reacher_ppo_1__20000000_1024_256_50_1.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl


python training_ppo.py     --project_name test --group_name first_run --exp_name reacher-ppo-numenvs256-numtimesteps20000000-batchsize1024-1-della-maxent-gaussianmlp- --num_evals 50     --seed 1 --num_timesteps 20000000 --batch_size 1024 --num_envs 256     --discounting 0.99 --action_repeat 1 --env_name reacher --use_dense_reward     --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 --log_wandb
  
