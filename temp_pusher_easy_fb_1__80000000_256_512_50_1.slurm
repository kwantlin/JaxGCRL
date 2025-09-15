#!/bin/bash

#SBATCH --job-name=pusher_easy_fb_1__80000000_256_512_50_1
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
#SBATCH --output=logs/della_pusher_easy_fb_1__80000000_256_512_50_1.out
#SBATCH --error=logs/della_pusher_easy_fb_1__80000000_256_512_50_1.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl


python training_fb.py     --project_name test --group_name first_run --exp_name pusher_easy-fb-della_1__80000000_256_512_50_1 --num_evals 50      --seed 1 --num_timesteps 80000000 --batch_size 256 --num_envs 512     --discounting 0.99 --action_repeat 1 --env_name pusher_easy     --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000     --train_step_multiplier 1 --log_wandb
