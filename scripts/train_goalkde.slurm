#!/bin/bash

#SBATCH -A lips
#SBATCH --gres=gpu:4  # Allocate 4 GPUs per node
#SBATCH -c 64         # Allocate 64 CPUs per node (16 per task * 4 tasks)
#SBATCH -N 2          # Request 2 nodes
#SBATCH --ntasks-per-node=4  # Run 4 tasks per node
#SBATCH --cpus-per-task=16   # 16 CPUs per task
#SBATCH -t 10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

# Function to run a single experiment
run_experiment() {
    local env=$1
    local seed=$2
    local var_post=$3
    local num_timesteps=$4
    local batch_size=$5
    local num_envs=$6
    local task_id=$7  # Add task_id parameter

    # Set CUDA environment variables
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES=$task_id  # Use task_id to assign unique GPU
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_CPP_MIN_LOG_LEVEL=2

    srun --gres=gpu:1 -c 16 --exclusive \
        python training_goalkde.py \
        --project_name test --group_name first_run --exp_name ${env}-goalkde${var_post:+-$var_post} --num_evals 50 \
        --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
        --discounting 0.99 --action_repeat 1 --env_name ${env} \
        --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
        --contrastive_loss_fn infonce_backward --energy_fn l2 \
        --train_step_multiplier 1 --log_wandb --var_post ${var_post:-standard} &
}

# Run experiments in parallel with task IDs
env=ant
run_experiment $env 1 standard 10000000 256 512 0
run_experiment $env 1 meanfield 10000000 256 512 1

env=simple_u_maze
run_experiment $env 1 standard 10000000 1024 256 2
run_experiment $env 1 meanfield 10000000 1024 256 3

env=reacher
run_experiment $env 1 standard 10000000 1024 256 4
run_experiment $env 1 meanfield 10000000 1024 256 5

env=pusher_easy
run_experiment $env 1 standard 30000000 1024 256 6
run_experiment $env 1 meanfield 30000000 1024 256 7

# Wait for all background processes to complete
wait

echo "All runs have finished."