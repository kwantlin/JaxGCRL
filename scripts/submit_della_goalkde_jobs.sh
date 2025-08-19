#!/bin/bash

# Function to submit a single job
submit_job() {
    local env=$1
    local seed=$2
    local var_post=$3
    local num_timesteps=$4
    local batch_size=$5
    local num_envs=$6
    local num_evals=${7:-50}
    local train_step_multiplier=${8:-1}
    local policy_lr=${9:-1e-4}
    local critic_lr=${10:-1e-4}
    local alpha_lr=${11:-1e-4}
    local notes=${9:-""}

    # Sanitize notes for filename and create a unique script name
    local sanitized_notes=${notes// /_}
    local slurm_script="temp_${env}_goalkde_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.slurm"

    # Create a temporary SLURM script for this job
    cat > ${slurm_script} << EOF
#!/bin/bash


#SBATCH --job-name=${env}_goalkde_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}_${policy_lr}_${critic_lr}_${alpha_lr}
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
#SBATCH --output=logs/della_${env}_goalkde_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}_${policy_lr}_${critic_lr}_${alpha_lr}.out
#SBATCH --error=logs/della_${env}_goalkde_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}_${policy_lr}_${critic_lr}_${alpha_lr}.err

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl

# Set CUDA environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

python training_goalkde.py \
    --project_name test --group_name first_run --exp_name ${env}-goalkde${var_post:+-$var_post}-${train_step_multiplier}x-${num_timesteps}-${batch_size}-${num_envs}-${policy_lr}-${critic_lr}-${alpha_lr}-${seed}-della-maxent-gaussianmlp-${notes} --num_evals ${num_evals} \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier ${train_step_multiplier} --log_wandb --var_post ${var_post:-standard} --policy_lr ${policy_lr} --critic_lr ${critic_lr} --alpha_lr ${alpha_lr}
EOF

    # Submit the job and run in background
    sbatch ${slurm_script} &
}

# Submit jobs for each environment
env=ant
submit_job $env 1 standard 60000000 256 512 50
submit_job $env 2 standard 60000000 256 512 50
submit_job $env 3 standard 60000000 256 512 50

submit_job $env 1 meanfield 60000000 256 512 50
submit_job $env 2 meanfield 60000000 256 512 50
submit_job $env 3 meanfield 60000000 256 512 50

submit_job $env 1 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4
submit_job $env 2 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4
submit_job $env 3 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4

submit_job $env 1 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4
submit_job $env 2 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4
submit_job $env 3 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4



env=simple_u_maze
# submit_job $env 1 standard 20000000 1024 256 50
# submit_job $env 1 meanfield 20000000 1024 256 50
# submit_job $env 1 meanfield_encoded 20000000 1024 256 50

env=reacher
# submit_job $env 1 standard 20000000 1024 256 50
# submit_job $env 1 meanfield 20000000 1024 256 50
# submit_job $env 1 meanfield_encoded 20000000 1024 256 50

env=pusher_easy
# submit_job $env 1 standard 60000000 256 512 
# submit_job $env 2 standard 60000000 256 512 
# submit_job $env 3 standard 60000000 256 512 
# submit_job $env 4 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 5 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 6 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 7 standard 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 8 standard 60000000 256 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 9 standard 60000000 256 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 10 standard 60000000 256 256 50 1 1e-4 1e-4 1e-4

# submit_job $env 1 meanfield 60000000 256 512 
# submit_job $env 2 meanfield 60000000 256 512 
# submit_job $env 3 meanfield 60000000 256 512 
# submit_job $env 4 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 5 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 6 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 7 meanfield 60000000 256 512 50 1 1e-4 1e-4 1e-4
# submit_job $env 8 meanfield 60000000 256 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 9 meanfield 60000000 256 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 10 meanfield 60000000 256 256 50 1 1e-4 1e-4 1e-4

# submit_job $env 1 meanfield_encoded 60000000 256 512 
# submit_job $env 2 meanfield_encoded 60000000 256 512 

# env=arm_reach
# submit_job $env 1 standard 12000000000 1024 2048 1000 8
# submit_job $env 1 meanfield 12000000000 1024 2048 1000 8
# submit_job $env 1 meanfield_encoded 12000000000 1024 2048 1000 8


# env=ant_fullobs
# submit_job $env 1 standard 12000000000 256 512 50
# submit_job $env 2 standard 12000000000 256 512 50
# submit_job $env 3 standard 12000000000 256 512 50


# submit_job $env 1 meanfield 12000000000 256 512 50
# submit_job $env 2 meanfield 12000000000 256 512 50
# submit_job $env 3 meanfield 12000000000 256 512 50



# env=ant_posvel
# submit_job $env 1 standard 12000000000 256 512 50
# submit_job $env 2 standard 12000000000 256 512 50
# submit_job $env 3 standard 12000000000 256 512 50


# submit_job $env 1 meanfield 12000000000 256 512 50
# submit_job $env 2 meanfield 12000000000 256 512 50
# submit_job $env 3 meanfield 12000000000 256 512 50


# submit_job $env 1 standard 1200000000 512 2048 1000
# submit_job $env 1 meanfield 1200000000 512 2048 1000
# submit_job $env 1 meanfield_encoded 1200000000 512 2048 1000






# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 