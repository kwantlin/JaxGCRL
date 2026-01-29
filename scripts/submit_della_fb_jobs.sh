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
    local notes=${8:-""}

    # Sanitize notes for filename and create a unique script name
    local sanitized_notes=${notes// /_}
    local slurm_script="temp_${env}_fb_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.slurm"

    # Create a temporary SLURM script for this job
    cat > ${slurm_script} << EOF
#!/bin/bash

#SBATCH --job-name=${env}_fb_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}
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
#SBATCH --output=logs/della_${env}_fb_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.out
#SBATCH --error=logs/della_${env}_fb_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier}.err

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl


python training_fb.py \
    --project_name test --group_name first_run --exp_name ${env}-fb-della_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}_${train_step_multiplier} --num_evals ${num_evals} \
     --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --train_step_multiplier ${train_step_multiplier} --log_wandb
EOF

    # Submit the job and run in background
    sbatch ${slurm_script} &
}

# Submit jobs for each environment
env=ant
submit_job $env 1 30000000 512 1024 100 1 "rebuttal"
submit_job $env 2 30000000 512 1024 100 1 "rebuttal"
submit_job $env 3 30000000 512 1024 100 1 "rebuttal"
submit_job $env 4 30000000 512 1024 100 1 "rebuttal"
submit_job $env 5 30000000 512 1024 100 1 "rebuttal"

env=simple_u_maze
submit_job $env 1 40000000 1024 256 50
submit_job $env 2 40000000 1024 256 50
submit_job $env 3 40000000 1024 256 50
submit_job $env 4 40000000 1024 256 50
submit_job $env 5 40000000 1024 256 50

env=reacher
submit_job $env 1 30000000 1024 256 100 1 "rebuttal"
submit_job $env 2 30000000 1024 256 100 1 "rebuttal"
submit_job $env 3 30000000 1024 256 100 1 "rebuttal"
submit_job $env 4 30000000 1024 256 100 1 "rebuttal"
submit_job $env 5 30000000 1024 256 100 1 "rebuttal"

env=pusher_easy
submit_job $env 1 60000000 256 512 100 1 "rebuttal"
submit_job $env 2 60000000 256 512 100 1 "rebuttal"
submit_job $env 3 60000000 256 512 100 1 "rebuttal"
submit_job $env 4 60000000 256 512 100 1 "rebuttal"
submit_job $env 5 60000000 256 512 100 1 "rebuttal"

# env=arm_reach
# submit_job $env 4 12000000000 1024 2048 1000 8

# submit_job $env 2 60000000 1024 1024 100
# submit_job $env 3 60000000 1024 1024 100
# submit_job $env 4 60000000 1024 1024 100
# submit_job $env 5 60000000 1024 1024 100

# env=ant_fullobs
# submit_job $env 2 12000000000 256 512 500 
# submit_job $env 3 12000000000 256 512 500  
# submit_job $env 4 12000000000 256 512 500 
# submit_job $env 5 12000000000 256 512 500 

env=ant_posvel
submit_job $env 1 120000000 256 512 500 1 "rebuttal"
submit_job $env 2 120000000 256 512 500 1 "rebuttal"
submit_job $env 3 120000000 256 512 500 1 "rebuttal"
submit_job $env 4 120000000 256 512 500 1 "rebuttal"
submit_job $env 5 120000000 256 512 500 1 "rebuttal"


env=ant_u_maze
# submit_job $env 1 30000000 512 1024 50 1 "rebuttal"

env=ant_angvel
submit_job $env 1 120000000 256 512 50 1 "rebuttal"
submit_job $env 2 120000000 256 512 50 1 "rebuttal"
submit_job $env 3 120000000 256 512 50 1 "rebuttal"
submit_job $env 4 120000000 256 512 50 1 "rebuttal"
submit_job $env 5 120000000 256 512 50 1 "rebuttal"




# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 