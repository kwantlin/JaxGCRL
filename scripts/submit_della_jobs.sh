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
    local notes=${9:-""}

    # Sanitize notes for filename and create a unique script name
    local sanitized_notes=${notes// /_}
    local slurm_script="temp_${env}_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.slurm"

    # Create a temporary SLURM script for this job
    cat > ${slurm_script} << EOF
#!/bin/bash

#SBATCH --job-name=${env}_main_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 16:00:00
#SBATCH --partition=pli 
#SBATCH --account=buildstuff
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu
#SBATCH --output=logs/della_${env}_main_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.out
#SBATCH --error=logs/della_${env}_main_${var_post}_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.err

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl



python training.py \
    --project_name test --group_name first_run --exp_name ${env}-main${var_post:+-$var_post}-numenvs${num_envs}-numtimesteps${num_timesteps}-batchsize${batch_size}-${train_step_multiplier}-della-maxent-gaussianmlp-${notes} --num_evals ${num_evals} \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier ${train_step_multiplier} --log_wandb --var_post ${var_post:-standard}
EOF

    # Submit the job and run in background
    sbatch ${slurm_script} &
}

# Submit jobs for each environment
# env=ant
# submit_job $env 1 standard 30000000 512 1024 50
# submit_job $env 2 standard 30000000 512 1024 50
# submit_job $env 3 standard 30000000 512 1024 50
# submit_job $env 4 standard 30000000 512 1024 50
# submit_job $env 5 standard 30000000 512 1024 50

# submit_job $env 1 meanfield 30000000 512 1024 50
# submit_job $env 2 meanfield 30000000 512 1024 50
# submit_job $env 3 meanfield 30000000 512 1024 50
# submit_job $env 4 meanfield 30000000 512 1024 50
# submit_job $env 5 meanfield 30000000 512 1024 50

env=simple_u_maze
# submit_job $env 1 standard 20000000 1024 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 2 standard 20000000 1024 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 3 standard 20000000 1024 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 4 standard 20000000 1024 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 5 standard 20000000 1024 256 50 1 1e-4 1e-4 1e-4

submit_job $env 1 meanfield 40000000 1024 256 50 1 1e-4 1e-4 1e-4
submit_job $env 2 meanfield 40000000 1024 256 50 1 1e-4 1e-4 1e-4
submit_job $env 3 meanfield 40000000 1024 256 50 1 1e-4 1e-4 1e-4
submit_job $env 4 meanfield 40000000 1024 256 50 1 1e-4 1e-4 1e-4
submit_job $env 5 meanfield 40000000 1024 256 50 1 1e-4 1e-4 1e-4

# env=simple_custom_maze
# submit_job $env 2 standard 40000000 1024 256 50 1 1e-4 1e-4 1e-4
# submit_job $env 2 meanfield 40000000 1024 256 50 1 1e-4 1e-4 1e-4

env=simple_big_maze
submit_job $env 2 standard 100000000 1024 256 50 1 1e-4 1e-4 1e-4
submit_job $env 2 meanfield 100000000 1024 256 50 1 1e-4 1e-4 1e-4

env=simple_hardest_maze
submit_job $env 2 standard 200000000 1024 256 50 1 1e-4 1e-4 1e-4
submit_job $env 2 meanfield 200000000 1024 256 50 1 1e-4 1e-4 1e-4

# env=reacher
# submit_job $env 1 standard 20000000 1024 256 50
# submit_job $env 2 standard 20000000 1024 256 50
# submit_job $env 3 standard 20000000 1024 256 50
# submit_job $env 4 standard 20000000 1024 256 50
# submit_job $env 5 standard 20000000 1024 256 50

# submit_job $env 1 meanfield 20000000 1024 256 50
# submit_job $env 2 meanfield 20000000 1024 256 50
# submit_job $env 3 meanfield 20000000 1024 256 50
# submit_job $env 4 meanfield 20000000 1024 256 50
# submit_job $env 5 meanfield 20000000 1024 256 50

# env=pusher_easy
# submit_job $env 1 standard 60000000 256 512 
# submit_job $env 2 standard 60000000 256 512 
# submit_job $env 3 standard 60000000 256 512 
# submit_job $env 4 standard 60000000 256 512 
# submit_job $env 5 standard 60000000 256 512 

# submit_job $env 1 meanfield 60000000 256 512 
# submit_job $env 2 meanfield 60000000 256 512 
# submit_job $env 3 meanfield 60000000 256 512 
# submit_job $env 4 meanfield 60000000 256 512 
# submit_job $env 5 meanfield 60000000 256 512 

# env=arm_reach
# submit_job $env 1 standard 12000000000 1024 2048 1000 8
# submit_job $env 1 meanfield 12000000000 1024 2048 1000 8
# submit_job $env 1 meanfield_encoded 12000000000 1024 2048 1000 8

env=ant_u_maze
# submit_job $env 1 standard 30000000 512 1024 50
# submit_job $env 1 meanfield 30000000 512 1024 50
# submit_job $env 1 meanfield_encoded 30000000 512 1024 50


# env=ant_fullobs
# submit_job $env 1 standard 12000000000 256 512 50
# submit_job $env 1 meanfield 12000000000 256 512 50



# env=ant_posvel
# submit_job $env 1 standard 12000000000 256 512 50
# submit_job $env 2 standard 12000000000 256 512 50
# submit_job $env 3 standard 12000000000 256 512 50
# submit_job $env 4 standard 12000000000 256 512 50
# submit_job $env 5 standard 12000000000 256 512 50
# submit_job $env 1 meanfield 12000000000 256 512 50
# submit_job $env 2 meanfield 12000000000 256 512 50
# submit_job $env 3 meanfield 12000000000 256 512 50
# submit_job $env 4 meanfield 12000000000 256 512 50
# submit_job $env 5 meanfield 12000000000 256 512 50

# env=ant_angvel
# submit_job $env 1 standard 120000000 256 512 50
# submit_job $env 1 meanfield 120000000 256 512 50
# submit_job $env 2 standard 120000000 256 512 50
# submit_job $env 2 meanfield 120000000 256 512 50
# submit_job $env 3 standard 120000000 256 512 50
# submit_job $env 3 meanfield 120000000 256 512 50
# submit_job $env 4 standard 120000000 256 512 50
# submit_job $env 4 meanfield 120000000 256 512 50
# submit_job $env 5 standard 120000000 256 512 50
# submit_job $env 5 meanfield 120000000 256 512 50


# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 