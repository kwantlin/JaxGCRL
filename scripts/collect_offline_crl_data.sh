#!/bin/bash

# python collect_offline_crl_data.py \
#     --run_dir /home/kw2960/JaxGCRL/runs/run_ant-main-standard-della-maxent-gaussianmlp_s_1 \
#     --ckpt_name best.pkl \
#     --output_path /scratch/gpfs/EYSENBACH/kw2960/JaxGCRL/data/ant_offline_crl_dataset.npz \
#     --num_envs 2000 \
#     --episode_length 1025 \
#     --seed 0 \
#     --deterministic


# python collect_offline_crl_data.py \
#     --run_dir /home/kw2960/JaxGCRL/runs/run_simple_u_maze-main-standard-della-maxent-gaussianmlp_s_1 \
#     --ckpt_name best.pkl \
#     --output_path /scratch/gpfs/EYSENBACH/kw2960/JaxGCRL/data/simple_u_maze_offline_crl_dataset.npz \
#     --num_envs 2000 \
#     --episode_length 1025 \
#     --seed 0 \
#     --deterministic


# python collect_offline_crl_data.py \
#     --run_dir /home/kw2960/JaxGCRL/runs/run_reacher-main-standard-della-maxent-gaussianmlp_s_1 \
#     --ckpt_name best.pkl \
#     --output_path /scratch/gpfs/EYSENBACH/kw2960/JaxGCRL/data/reacher_offline_crl_dataset.npz \
#     --num_envs 2000 \
#     --episode_length 1025 \
#     --seed 0 \
#     --deterministic

python collect_offline_crl_data.py \
    --run_dir /home/kw2960/JaxGCRL/runs/run_pusher_easy-main-meanfield-numenvs512-numtimesteps60000000-batchsize256-1-della-maxent-gaussianmlp-_s_3 \
    --ckpt_name best.pkl \
    --output_path /scratch/gpfs/EYSENBACH/kw2960/JaxGCRL/data/pusher_easy_offline_crl_dataset.npz \
    --num_envs 2000 \
    --episode_length 1025 \
    --seed 0 \
    --deterministic