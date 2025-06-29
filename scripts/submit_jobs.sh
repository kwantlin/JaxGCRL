#!/bin/bash

# Function to submit a single job
submit_job() {
    local env=$1
    local seed=$2
    local var_post=$3
    local num_timesteps=$4
    local batch_size=$5
    local num_envs=$6

    # Create a temporary SLURM script for this job
    cat > temp_${env}_${var_post}.slurm << EOF
#!/bin/bash

#SBATCH -A lips
#SBATCH --job-name=${env}_${var_post}
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH -t 26:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl



python training.py \
    --project_name test --group_name first_run --exp_name ${env}-main${var_post:+-$var_post} --num_evals 50 \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier 1 --log_wandb --var_post ${var_post:-standard}
EOF

    # Submit the job and run in background
    sbatch temp_${env}_${var_post}.slurm &
}

# Submit jobs for each environment
env=ant
submit_job $env 1 standard 20000000 256 512
submit_job $env 1 meanfield 20000000 256 512

env=simple_u_maze
submit_job $env 1 standard 20000000 1024 256
submit_job $env 1 meanfield 20000000 1024 256

env=reacher
submit_job $env 1 standard 20000000 1024 256
submit_job $env 1 meanfield 20000000 1024 256

env=pusher_easy
submit_job $env 1 standard 60000000 1024 256
submit_job $env 1 meanfield 60000000 1024 256

# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 