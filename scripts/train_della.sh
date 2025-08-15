#!/bin/bash

# Function to submit a single job
submit_job() {
    local env=$1
    local seed=$2
    local num_timesteps=$3
    local batch_size=$4
    local num_envs=$5
    local num_evals=$6
    local train_step_multiplier=${7:-1}
    local episode_length=${8:-1000}
    local notes=${9:-""}
    

    # Sanitize notes for filename and create a unique script name
    local sanitized_notes=${notes// /_}
    local slurm_script="temp_${env}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.slurm"

    # Create a temporary SLURM script for this job
    cat > ${slurm_script} << EOF
#!/bin/bash

#SBATCH --job-name=${env}_main_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}
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
#SBATCH --output=logs/della_${env}_main_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.out
#SBATCH --error=logs/della_${env}_main_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.err

eval "\$(conda shell.bash hook)"
conda activate jaxgcrlsudoku



XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python main.py crl \
    --wandb_project_name sudoku-bmm --wandb_group first_run --exp_name ${env}_${seed}_${num_timesteps}_${batch_size}_${num_envs}_${train_step_multiplier}_${notes} --num_evals ${num_evals} \
    --seed ${seed} --total_env_steps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env ${env} \
    --episode_length ${episode_length} --unroll_length ${episode_length} --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn bwd_infonce --energy_fn norm \
    --train_step_multiplier ${train_step_multiplier}
EOF

    # Submit the job and run in background
    sbatch ${slurm_script} &
}

# Submit jobs for each environment
# env=arm_push_easy
# submit_job $env 1 800000000 256 512 50 64 "largenet" 
# submit_job $env 2 800000000 256 512 50 64 "largenet"
# submit_job $env 3 800000000 256 512 50 64 "largenet"
# submit_job $env 4 800000000 256 512 50 64 "largenet"
# submit_job $env 5 800000000 256 512 50 64 "largenet"

env=sudoku
submit_job $env 1 800000000 256 512 50 4 81 "largenet" 
submit_job $env 2 800000000 256 512 50 4 81 "largenet"
submit_job $env 3 800000000 256 512 50 4 81 "largenet"
submit_job $env 4 800000000 256 512 50 4 81 "largenet"
submit_job $env 5 800000000 256 512 50 4 81 "largenet"


wait

echo "All jobs have been submitted." 