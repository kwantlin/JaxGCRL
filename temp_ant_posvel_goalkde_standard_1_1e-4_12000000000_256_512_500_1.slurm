#!/bin/bash


#SBATCH --job-name=ant_posvel_goalkde_standard_1_1e-4_12000000000_256_512_500_1_1e-4_1e-4_1e-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -t 30:00:00
#SBATCH --partition=pli 
#SBATCH --account=buildstuff
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu
#SBATCH --output=logs/della_ant_posvel_goalkde_standard_1_1e-4_12000000000_256_512_500_1_1e-4_1e-4_1e-4.out
#SBATCH --error=logs/della_ant_posvel_goalkde_standard_1_1e-4_12000000000_256_512_500_1_1e-4_1e-4_1e-4.err

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

# Set CUDA environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

python training_goalkde.py     --project_name test --group_name first_run --exp_name ant_posvel-goalkde-standard-1x-12000000000-256-512-1e-4-1e-4-1e-4-1-della-maxent-gaussianmlp-1e-4 --num_evals 500     --seed 1 --num_timesteps 12000000000 --batch_size 256 --num_envs 512     --discounting 0.99 --action_repeat 1 --env_name ant_posvel     --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000     --contrastive_loss_fn infonce_backward --energy_fn l2     --train_step_multiplier 1 --log_wandb --var_post standard --policy_lr 1e-4 --critic_lr 1e-4 --alpha_lr 1e-4
