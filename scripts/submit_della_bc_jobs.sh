#!/bin/bash

# Function to submit a single job
submit_job() {
    local env=$1
    local seed=$2
    local var_post=$3
    local num_timesteps=$4
    local batch_size=$5
    local num_envs=$6
    local saved_ckpt_path=$7
    local num_evals=$8
    local train_step_multiplier=${9:-1}
    local notes=${10:-""}

    # Sanitize notes for filename and create a unique script name
    local sanitized_notes=${notes// /_}
    local slurm_script="temp_bc_${env}_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.slurm"

    # Create a temporary SLURM script for this job
    cat > ${slurm_script} << EOF
#!/bin/bash

#SBATCH --job-name=${env}_bc_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}
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
#SBATCH --output=logs/della_${env}_bc_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.out
#SBATCH --error=logs/della_${env}_bc_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.err

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl

# Set CUDA environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

python training_bc.py \
    --project_name test --group_name first_run --exp_name ${env}-bc${var_post:+-$var_post}-${train_step_multiplier}x-${num_timesteps}-${batch_size}-${num_envs}-${num_evals} --num_evals ${num_evals} \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier ${train_step_multiplier} --saved_ckpt_path ${saved_ckpt_path} --log_wandb --disable_entropy_actor \
    --var_post ${var_post:-standard}
EOF

    # Submit the job and run in background
    sbatch ${slurm_script} &
}

# Submit jobs for each environment
# env=ant
# saved_ckpt_path="/home/kw2960/JaxGCRL/runs/run_ant-main-standard-della-maxent-gaussianmlp_s_1/ckpt/best.pkl"
# submit_job $env 1 standard 30000000 512 1024 "$saved_ckpt_path" 50
# submit_job $env 1 meanfield 30000000 512 1024 "$saved_ckpt_path" 50

# env=simple_u_maze 
# saved_ckpt_path="/home/kw2960/JaxGCRL/runs/run_simple_u_maze-main-standard-della-maxent-gaussianmlp_s_1/ckpt/best.pkl"
# submit_job $env 1 standard 20000000 1024 256 "$saved_ckpt_path" 50
# submit_job $env 1 meanfield 20000000 1024 256 "$saved_ckpt_path" 50

# env=reacher
# saved_ckpt_path="/home/kw2960/JaxGCRL/runs/run_reacher-main-standard-della-maxent-gaussianmlp_s_1/ckpt/best.pkl"
# submit_job $env 1 standard 20000000 1024 256 "$saved_ckpt_path" 50
# submit_job $env 1 meanfield 20000000 1024 256 "$saved_ckpt_path" 50

# env=pusher_easy
# saved_ckpt_path="/home/kw2960/JaxGCRL/runs/run_pusher_easy-main-standard-numenvs2048-numtimesteps60000000-batchsize1024-della-maxent-gaussianmlp-_s_1/ckpt/best.pkl"
# submit_job $env 1 standard 60000000 1024 1024 "$saved_ckpt_path" 100
# submit_job $env 1 meanfield 60000000 1024 1024 "$saved_ckpt_path" 100

env=arm_reach
saved_ckpt_path="/home/kw2960/JaxGCRL/runs/run_arm_reach-main-standard-numenvs2048-numtimesteps60000000-batchsize1024-della-maxent-gaussianmlp-_s_1/ckpt/best.pkl"
submit_job $env 1 standard 12000000000 1024 2048 "$saved_ckpt_path" 1000 16
submit_job $env 1 meanfield 12000000000 1024 2048 "$saved_ckpt_path" 1000 16


# env=ant_fullobs
# saved_ckpt_path="/scratch/gpfs/kw2960/JaxGCRL/runs/run_ant_fullobs-main-standard-della-maxent-gaussianmlp-more_envs_s_1/ckpt/best.pkl"
# submit_job $env 1 standard 1200000000 512 2048 "$saved_ckpt_path" 100
# submit_job $env 1 meanfield 1200000000 512 2048 "$saved_ckpt_path" 100


# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 