#!/bin/bash

# Function to submit a single job
submit_job() {
    local env=$1
    local seed=$2
    local num_timesteps=$3
    local batch_size=$4
    local num_envs=$5
    local num_evals=${6:-50}
    local notes=${7:-""}

    # Sanitize notes for filename and create a unique script name
    local sanitized_notes=${notes// /_}
    local slurm_script="temp_${env}_ppo_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}.slurm"

    # Create a temporary SLURM script for this job
    cat > ${slurm_script} << EOF
#!/bin/bash

#SBATCH --job-name=${env}_ppo_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --partition=pli 
#SBATCH --account=buildstuff
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu
#SBATCH --output=logs/della_${env}_ppo_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}.out
#SBATCH --error=logs/della_${env}_ppo_${seed}_${sanitized_notes}_${num_timesteps}_${batch_size}_${num_envs}_${num_evals}.err

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl


python training_ppo.py \
    --project_name test --group_name first_run --exp_name ${env}-ppo-numenvs${num_envs}-numtimesteps${num_timesteps}-batchsize${batch_size}-della-maxent-gaussianmlp-${notes} --num_evals ${num_evals} \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} --use_dense_reward \
    --episode_length 1025 --unroll_length 62  --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 --log_wandb
  
EOF

    # Submit the job and run in background
    sbatch ${slurm_script} &
}

# Submit jobs for each environment
env=ant
submit_job $env 1 3000000000 512 1024 100 "newpponet" 


# env=simple_u_maze
# submit_job $env 1 20000000 1024 256 50


env=reacher
submit_job $env 1 2000000000 1024 256 100 "newpponet" 

env=pusher_easy
submit_job $env 1 6000000000 256 512 100 "newpponet" 


# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 