#!/bin/bash

#SBATCH --job-name=ant_posvel_fb_3__12000000000_256_512_500_1
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
#SBATCH --output=logs/della_ant_posvel_fb_3__12000000000_256_512_500_1.out
#SBATCH --error=logs/della_ant_posvel_fb_3__12000000000_256_512_500_1.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl


python training_fb.py     --project_name test --group_name first_run --exp_name ant_posvel-fb-della_3__12000000000_256_512_500_1 --num_evals 500      --seed 3 --num_timesteps 12000000000 --batch_size 256 --num_envs 512     --discounting 0.99 --action_repeat 1 --env_name ant_posvel     --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000     --train_step_multiplier 1 --log_wandb
