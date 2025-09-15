#!/bin/bash

#SBATCH --job-name=pusher_easy_ppo_1_long_training_60000000000_256_512_100
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
#SBATCH --output=logs/della_pusher_easy_ppo_1_long_training_60000000000_256_512_100.out
#SBATCH --error=logs/della_pusher_easy_ppo_1_long_training_60000000000_256_512_100.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl


python training_ppo.py     --project_name test --group_name first_run --exp_name pusher_easy-ppo-numenvs512-numtimesteps60000000000-batchsize256-della-maxent-gaussianmlp-long_training --num_evals 100     --seed 1 --num_timesteps 60000000000 --batch_size 256 --num_envs 512     --discounting 0.99 --action_repeat 1 --env_name pusher_easy --use_dense_reward     --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 --log_wandb
  
