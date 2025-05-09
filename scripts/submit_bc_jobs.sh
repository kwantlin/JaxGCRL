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

    # Create a temporary SLURM script for this job
    cat > temp_${env}_bc_${var_post}.slurm << EOF
#!/bin/bash

#SBATCH -A lips
#SBATCH --job-name=${env}_bc_${var_post}
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 6:00:00

eval "\$(conda shell.bash hook)"
conda activate jaxgcrl

# Set CUDA environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

python training_bc.py \
    --project_name test --group_name first_run --exp_name ${env}-bc${var_post:+-$var_post} --num_evals 50 \
    --seed ${seed} --num_timesteps ${num_timesteps} --batch_size ${batch_size} --num_envs ${num_envs} \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1025 --unroll_length 62 --n_hidden 8 --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --train_step_multiplier 1 --saved_ckpt_path ${saved_ckpt_path} --log_wandb --disable_entropy_actor \
    --var_post ${var_post:-standard}
EOF

    # Submit the job and run in background
    sbatch temp_${env}_bc_${var_post}.slurm &
}

# Submit jobs for each environment
# env=ant
# saved_ckpt_path="/n/fs/klips/JaxGCRL/runs/run_ant-main-standard_s_1/ckpt/step_11427840.pkl"
# submit_job $env 1 standard 10000000 256 512 "$saved_ckpt_path"
# submit_job $env 1 meanfield 10000000 256 512 "$saved_ckpt_path"

# env=simple_u_maze 
# saved_ckpt_path="/n/fs/klips/JaxGCRL/runs/run_simple_u_maze-main-standard_s_1/ckpt/step_20490752.pkl"
# submit_job $env 1 standard 20000000 1024 256 "$saved_ckpt_path"
# submit_job $env 1 meanfield 20000000 1024 256 "$saved_ckpt_path"

# env=reacher
# saved_ckpt_path="/n/fs/klips/JaxGCRL/runs/run_reacher-main-standard_s_1/ckpt/step_20490752.pkl"
# submit_job $env 1 standard 20000000 1024 256 "$saved_ckpt_path"
# submit_job $env 1 meanfield 20000000 1024 256 "$saved_ckpt_path"

env=pusher_easy
saved_ckpt_path="/n/fs/klips/JaxGCRL/runs/run_pusher_easy-main-standard_s_1/ckpt/step_39378432.pkl"
submit_job $env 1 standard 40000000 1024 256 "$saved_ckpt_path"
submit_job $env 1 meanfield 40000000 1024 256 "$saved_ckpt_path"

# Wait for all background processes to complete
wait

echo "All jobs have been submitted." 