# Script to evaluate legibility of Diffusion Policy with and without the legibility module

import os
import torch
import numpy as np
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils.architecture_utils import ControlConditionalUnet1D, SceneEncoder, FiLMGenerator, ConfigurableFiLMGenerator
from utils.dataset_utils import normalize_data, unnormalize_data
from tqdm import tqdm
import sys
import pickle
import torch.nn as nn
import panda_py
import pandas as pd
import time
import random
from scipy.interpolate import interp1d

from utils.inference_utils_t import rollout_to_goal, rollout_to_goal_legidiff

def resample_trajectory(trajectory, target_steps=100):
    """
    Interpolates a trajectory to a fixed number of steps 
    preserving the geometric shape.
    """
    trajectory = np.array(trajectory)
    # Original time indices (0, 1, 2, ... N)
    original_steps = np.linspace(0, 1, len(trajectory))
    
    # Target time indices (0, 0.01, 0.02 ... 1.0)
    target_step_indices = np.linspace(0, 1, target_steps)
    
    # Create an interpolator function for each dimension (x, y, etc.)
    # axis=0 means we interpolate along the rows (time steps)
    interpolator = interp1d(original_steps, trajectory, axis=0, kind='linear')
    
    return interpolator(target_step_indices)

goal_configurations = [
    (1, [2.0, -0.2], [2.0, 0.2]),
    (2, [1.55, -0.4], [1.6, -0.2]),
    (3, [2.2, 0.1], [2.1, 0.4]),
    (4, [2.0, -0.9], [0.9, 0.5]), 
    (5, [1.8, -0.7], [1.8, 0.7]),
    (6, [1.0, -0.2], [2.2, 0.9]),
]

for i in range(4, 7):

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    device = torch.device('cuda')

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    obs_dim = 5 
    obs_dim_legdiff = 5
    action_dim = 3

    noise_pred_net = ControlConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
    )

    noise_pred_net_legdiff = ControlConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim_legdiff*obs_horizon
    )

    ckpt_path = f"output/ckpt/t_dp_{i}.ckpt"
    if not os.path.isfile(ckpt_path):
        exit("Checkpoint file not found.")
        
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state_dict["model_state_dict"]
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net = ema_noise_pred_net.to(device)
    ema_noise_pred_net.load_state_dict(state_dict, strict=False)

    ckpt_path_legdiff = f"output/ckpt/t_legidiff_{i}.ckpt"
    if not os.path.isfile(ckpt_path):
        exit("Checkpoint file not found.")
        
    state_dict = torch.load(ckpt_path_legdiff, map_location=device, weights_only=False)
    state_dict = state_dict["model_state_dict"]
    ema_noise_pred_net_legdiff = noise_pred_net_legdiff
    ema_noise_pred_net_legdiff = ema_noise_pred_net_legdiff.to(device)
    ema_noise_pred_net_legdiff.load_state_dict(state_dict)
    print('Pretrained weights loaded.')

    encoder = SceneEncoder(
        obj_dim=7, num_objects=1
    ).to(device)
    encoder.load_state_dict(torch.load('output/scene_encoder/t_se_20000_50e.pth'))
    encoder.eval()

    # l_film_generator = FiLMGenerator(
    #     context_dim=64, feature_dims={4: 1024}
    # ).to(device)
    # l_film_generator.load_state_dict(torch.load(f'output/mlp/t_legibility_2000e_{i}.pt'))
    # l_film_generator.eval()
    # l_film_generator = FiLMGenerator(
    #     context_dim=64, feature_dims={4: 1024}
    # ).to(device)
    l_film_generator = ConfigurableFiLMGenerator(
        context_dim=64,
        feature_dims={4:1024},
        hidden_dim=1024,
        num_layers=4
    ).to(device)
    l_film_generator.load_state_dict(torch.load(f'output/mlp/t_predictability_300e_{i}.pt'))
    l_film_generator.eval()

    ema_noise_pred_net.eval()
    ema_noise_pred_net_legdiff.eval()

    max_steps = 200

    initial_joint = [0.0, 0.0, 0.0] 

    with open(f"stats/t_stats_{i}.pkl", "rb") as f:
        stats = pickle.load(f)
    with open(f"stats/t_stats_{i}_ld.pkl", "rb") as f:
        stats_ld = pickle.load(f)

    _, g1_coords, g2_coords = goal_configurations[i-1]
    # Convert lists to numpy arrays
    config_g1 = np.array(g1_coords)
    config_g2 = np.array(g2_coords)
    print(f"--- Processing Seed {i} ---")
    print(f"Goal Config: {config_g1} and {config_g2}")

    trajectories_without = []
    trajectories_l_film = []
    trajectories_p_film = []
    trajectories_ld = []

    for j in range(6):

        if random.uniform(0, 1) < 0.5:
            goal_1 = config_g1
            goal_2 = config_g2
        else:
            goal_1 = config_g2
            goal_2 = config_g1

        traj_l_film, success_l_film = rollout_to_goal(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, encoder=encoder, film=l_film_generator, method="film")

        traj_without, success_without = rollout_to_goal(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, method=None)

        traj_ld, success_ld = rollout_to_goal_legidiff(
        initial_joint, goal_1, goal_2, ema_noise_pred_net_legdiff, noise_scheduler, stats_ld)

        if (success_without == True) and (success_ld == True) and (success_l_film == True):

            traj_l_film_norm = resample_trajectory(traj_l_film, target_steps=200)
            traj_without_norm = resample_trajectory(traj_without, target_steps=200)
            traj_ld_norm = resample_trajectory(traj_ld, target_steps=200)

            trajectories_without.append(traj_without_norm)
            trajectories_l_film.append(traj_l_film_norm)
            trajectories_ld.append(traj_ld_norm)

        else:
            if success_without == False:
                print("Diffusion Policy: no success.")
            if success_ld == False:
                print("Legibility Diffuser: no success.")
            if success_l_film == False:
                print("DP + Legibility: no success.")

        import matplotlib.pyplot as plt

        # 1. Organize your data into a list of tuples (trajectory_list, label, color)
        all_trajectories = [
            (trajectories_without, "Diffusion Policy", "blue"),
            (trajectories_l_film, "SCDP", "blue"),
            (trajectories_ld, "Legibility Diffuser", "blue")
        ]

        print(goal_1)
        # 2. Iterate through each category
        for traj_list, label_name, color_code in all_trajectories:
            # Iterate through each specific trajectory in that category
            for i, traj in enumerate(traj_list):
                # Create a NEW figure for every single trajectory
                fig, ax = plt.subplots(figsize=(8, 6))
                
                traj_np = np.array(traj)
                ax.plot(
                    traj_np[:50, 0], traj_np[:50, 1],
                    color=color_code, linewidth=2
                )

                # Points de départ et d’arrivée (Goals)
                ax.scatter(config_g1[0], config_g1[1], color='orange', marker='*', s=150, label='Goal 1')
                ax.scatter(config_g2[0], config_g2[1], color='green', marker='*', s=150, label='Goal 2')

                # Formatting each individual plot
                ax.set_title(f"Trajectory: {label_name}", fontsize=14)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_xlim(-0.1, 2.5)
                ax.set_ylim(-1.2, 1.2)
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                # This will pop up a window for each trajectory
                plt.savefig(f"traj_pics/{j}_{label_name}_{goal_1[0]}_{goal_1[1]}_50.png")

        for traj_list, label_name, color_code in all_trajectories:
            # Iterate through each specific trajectory in that category
            for i, traj in enumerate(traj_list):
                # Create a NEW figure for every single trajectory
                fig, ax = plt.subplots(figsize=(8, 6))
                
                traj_np = np.array(traj)
                ax.plot(
                    traj_np[:100, 0], traj_np[:100, 1],
                    color=color_code, linewidth=2
                )

                # Points de départ et d’arrivée (Goals)
                ax.scatter(config_g1[0], config_g1[1], color='orange', marker='*', s=150, label='Goal 1')
                ax.scatter(config_g2[0], config_g2[1], color='green', marker='*', s=150, label='Goal 2')

                # Formatting each individual plot
                ax.set_title(f"Trajectory: {label_name}", fontsize=14)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_xlim(-0.1, 2.5)
                ax.set_ylim(-1.2, 1.2)
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                # This will pop up a window for each trajectory
                plt.savefig(f"traj_pics/{j}_{label_name}_{goal_1[0]}_{goal_1[1]}_100.png")

        for traj_list, label_name, color_code in all_trajectories:
            # Iterate through each specific trajectory in that category
            for i, traj in enumerate(traj_list):
                # Create a NEW figure for every single trajectory
                fig, ax = plt.subplots(figsize=(8, 6))
                
                traj_np = np.array(traj)
                ax.plot(
                    traj_np[:150, 0], traj_np[:150, 1],
                    color=color_code, linewidth=2
                )

                # Points de départ et d’arrivée (Goals)
                ax.scatter(config_g1[0], config_g1[1], color='orange', marker='*', s=150, label='Goal 1')
                ax.scatter(config_g2[0], config_g2[1], color='green', marker='*', s=150, label='Goal 2')

                # Formatting each individual plot
                ax.set_title(f"Trajectory: {label_name}", fontsize=14)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_xlim(-0.1, 2.5)
                ax.set_ylim(-1.2, 1.2)
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                # This will pop up a window for each trajectory
                plt.savefig(f"traj_pics/{j}_{label_name}_{goal_1[0]}_{goal_1[1]}_150.png")