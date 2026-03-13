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

from utils.inference_utils import rollout_to_goal
from utils.inference_utils_ld import rollout_to_goal_legdiff


def compute_distance_to_goal2(trajectory, goal_2):
    positions = np.array([list(pose[:3]) for pose in trajectory])

    distances = np.linalg.norm(positions - goal_2, axis=1)
    
    times = np.arange(1, len(trajectory) + 1)

    weighted_distances = distances / times

    return np.sum(weighted_distances)# /len(trajectory)

def compute_detachment_score(trajectory, goal_1, goal_2):
    """
    Computes the normalized detachment score: 
    L_norm = sum( (||g- - st|| / ||g* - g-||) / t )
    """
    positions = np.array([list(pose[:3]) for pose in trajectory])
    goal_dist = np.linalg.norm(np.array(goal_1[:3]) - np.array(goal_2[:3]))
    normalization_factor = max(goal_dist, 1e-6) 
    dist_to_neg_goal = np.linalg.norm(positions - goal_2[:3], axis=1)
    times = np.arange(1, len(trajectory) + 1)
    normalized_weighted_distances = (dist_to_neg_goal / normalization_factor) / times
    
    return np.sum(normalized_weighted_distances)

def compute_trajectory_length(trajectory):
    length = 0
    last_waypoint = None

    positions = np.array([list(pose[:3]) for pose in trajectory])

    for waypoint in positions:
        if last_waypoint is not None:
            dist = np.linalg.norm(waypoint - last_waypoint)
            length = length + dist

        last_waypoint = waypoint

    return length

def compute_trajectory_efficiency(trajectory, epsilon=1e-6):
    """
    Computes the trajectory efficiency as the reciprocal of the total 
    Euclidean distance traveled.
    """
    positions = np.array([pose[:3] for pose in trajectory])
    diffs = np.diff(positions, axis=0)

    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)

    return 1.0 / (total_length + epsilon)

def resample_trajectory(trajectory, target_steps=150):
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

global_stats = {
    'success': {'without': [], 'l_film': [], 'p_film': [], 'ld': []},
    'dist':    {'without': [], 'l_film': [], 'p_film': [], 'ld': []},
    'len':     {'without': [], 'l_film': [], 'p_film': [], 'ld': []}
}

goal_configurations = [
    (1, [0.5, -0.1, 0.1], [0.5, 0.1, 0.1]), # Spatial ambiguity
    (2, [0.55, -0.2, 0.1], [0.55, -0.1, 0.1]),
    (3, [0.55, 0.15, 0.1], [0.58, 0.3, 0.1]),
    (4, [0.5, -0.1, 0.1], [0.65, 0.3, 0.1]), # No spatial ambiguity
    (5, [0.4, -0.3, 0.1], [0.6, 0.3, 0.1]),
    (6, [0.4, 0.3, 0.1], [0.4, -0.3, 0.1]),
]

for i in range(1, 7):

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

    obs_dim = 10 
    obs_dim_legdiff = 10
    action_dim = 7

    noise_pred_net = ControlConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
    )

    noise_pred_net_legdiff = ControlConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim_legdiff*obs_horizon
    )

    ckpt_path = f"output/ckpt/f_dp_{i}.ckpt"
    if not os.path.isfile(ckpt_path):
        exit("Checkpoint file not found.")
        
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state_dict["model_state_dict"]
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net = ema_noise_pred_net.to(device)
    ema_noise_pred_net.load_state_dict(state_dict, strict=False)

    ckpt_path_legdiff = f"output/ckpt/f_legidiff_{i}.ckpt"
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
    encoder.load_state_dict(torch.load('output/scene_encoder/se_20000_50e.pth'))
    encoder.eval()

    # l_film_generator = FiLMGenerator(
    #     context_dim=64, feature_dims={4: 1024}
    # ).to(device)
    l_film_generator = ConfigurableFiLMGenerator(
        context_dim=64,
        feature_dims={4:1024},
        hidden_dim=1024,
        num_layers=4
    ).to(device)
    l_film_generator.load_state_dict(torch.load(f'output/mlp/f_legibility_300e_{i}.pt'))
    l_film_generator.eval()

    # p_film_generator = FiLMGenerator(
    #     context_dim=64, feature_dims={4: 1024}
    # ).to(device)
    p_film_generator = ConfigurableFiLMGenerator(
        context_dim=64,
        feature_dims={4:1024},
        hidden_dim=1024,
        num_layers=4
    ).to(device)
    p_film_generator.load_state_dict(torch.load(f'output/mlp/f_predictability_300e_{i}.pt'))
    p_film_generator.eval()

    ema_noise_pred_net.eval()
    ema_noise_pred_net_legdiff.eval()

    max_steps = 150

    initial_joint = [0.00015335381064485176, -0.7855436163999139, -8.498953462776626e-05, -2.3561257643176736, -6.181224792900508e-05, 1.5713097735372683, 0.7854071770339326]  # position de départ fixe

    with open(f"stats/f_stats_{i}.pkl", "rb") as f:
        stats = pickle.load(f)
    with open(f"stats/f_stats_{i}_ld.pkl", "rb") as f:
        stats_ld = pickle.load(f)

    _, g1_coords, g2_coords = goal_configurations[i-1]
    # Convert lists to numpy arrays
    config_g1 = np.array(g1_coords)
    config_g2 = np.array(g2_coords)
    print(f"--- Processing Seed {i} ---")
    print(f"Goal Config: {config_g1} and {config_g2}")

    dists_without_latent = []
    dists_ld = []
    dists_l_film = []
    dists_p_film = []

    successes_l_film = []
    successes_p_film = []
    successes_without_latent = []
    successes_leg_diff = []

    inference_times_with_half_latent = []
    inference_times_with_latent = []
    inference_times_without_latent = []
    inference_times_leg_diff = []

    lengths_ld = []
    lengths_l_film = []
    lengths_p_film = []
    lengths_without_latent = []

    with open(f"stats/f_legibility_bounds_{i}.txt", "r") as f:
        min_dist = float(f.readline().strip())
        max_dist = float(f.readline().strip())

    # with open(f"stats/f_length_bounds_{i}.txt") as f:
    #     min_length = float(f.readline().strip())
    #     max_length = float(f.readline().strip())

    with open(f"stats/f_eff_bounds_{i}.txt") as f:
        min_length = float(f.readline().strip())
        max_length = float(f.readline().strip())

    for j in range(30):

        if random.uniform(0, 1) < 0.5:
            goal_1 = config_g1
            goal_2 = config_g2
        else:
            goal_1 = config_g2
            goal_2 = config_g1

        traj_l_film, success_l_film = rollout_to_goal(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, encoder=encoder, film=l_film_generator, method="film")

        traj_p_film, success_p_film = rollout_to_goal(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, encoder=encoder, film=p_film_generator, method="film")

        traj_without, success_without = rollout_to_goal(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, method=None)

        traj_ld, success_ld = rollout_to_goal_legdiff(
        initial_joint, goal_1, goal_2, ema_noise_pred_net_legdiff, noise_scheduler, stats_ld)

        successes_l_film.append(success_l_film)
        successes_p_film.append(success_p_film)
        successes_without_latent.append(success_without)
        successes_leg_diff.append(success_ld)

        if (success_without == True) and (success_ld == True) and (success_l_film == True) and (success_p_film == True):

            traj_l_film_norm = resample_trajectory(traj_l_film, target_steps=200)
            traj_p_film_norm = resample_trajectory(traj_p_film, target_steps=200)
            traj_without_norm = resample_trajectory(traj_without, target_steps=200)
            traj_ld_norm = resample_trajectory(traj_ld, target_steps=200)

            # dists_l_film.append(compute_distance_to_goal2(traj_l_film_norm, goal_2))
            # dists_p_film.append(compute_distance_to_goal2(traj_p_film_norm, goal_2))
            # dists_without_latent.append(compute_distance_to_goal2(traj_without_norm, goal_2))
            # dists_ld.append(compute_distance_to_goal2(traj_ld_norm, goal_2))
            dists_l_film.append(compute_detachment_score(traj_l_film_norm, goal_1, goal_2))
            dists_p_film.append(compute_detachment_score(traj_p_film_norm, goal_1, goal_2))
            dists_without_latent.append(compute_detachment_score(traj_without_norm, goal_1, goal_2))
            dists_ld.append(compute_detachment_score(traj_ld_norm, goal_1, goal_2))

            # lengths_l_film.append(compute_trajectory_length(traj_l_film))
            # lengths_p_film.append(compute_trajectory_length(traj_p_film))
            # lengths_without_latent.append(compute_trajectory_length(traj_without))
            # lengths_ld.append(compute_trajectory_length(traj_ld))

            lengths_l_film.append(compute_trajectory_efficiency(traj_l_film))
            lengths_p_film.append(compute_trajectory_efficiency(traj_p_film))
            lengths_without_latent.append(compute_trajectory_efficiency(traj_without))
            lengths_ld.append(compute_trajectory_efficiency(traj_ld))

        else:
            if success_without == False:
                print("Diffusion Policy: no success.")
            if success_ld == False:
                print("Legibility Diffuser: no success.")
            if success_l_film == False:
                print("DP + Legibility: no success.")
            if success_p_film == False:
                print("DP + Predictability: no success.")

    normalized_scores_ld = [(d - min_dist) / (max_dist - min_dist) for d in dists_ld]
    normalized_scores_l_film = [(d - min_dist) / (max_dist - min_dist) for d in dists_l_film]
    normalized_scores_p_film = [(d - min_dist) / (max_dist - min_dist) for d in dists_p_film]
    normalized_scores_without = [(d - min_dist) / (max_dist - min_dist) for d in dists_without_latent]

    normalized_lengths_ld = [(l - min_length) / (max_length - min_length) for l in lengths_ld]
    normalized_lengths_l_film = [(l - min_length) / (max_length - min_length) for l in lengths_l_film]
    normalized_lengths_p_film = [(l - min_length) / (max_length - min_length) for l in lengths_p_film]
    normalized_lengths_without = [(l - min_length) / (max_length - min_length) for l in lengths_without_latent]

    global_stats['success']['without'].append(np.mean(successes_without_latent))
    global_stats['success']['l_film'].append(np.mean(successes_l_film))
    global_stats['success']['p_film'].append(np.mean(successes_p_film))
    global_stats['success']['ld'].append(np.mean(successes_leg_diff)) 

    # 2. Store Normalized Distances (Mean of the successful steps for this iteration)
    global_stats['dist']['without'].append(np.mean(normalized_scores_without))
    global_stats['dist']['l_film'].append(np.mean(normalized_scores_l_film))
    global_stats['dist']['p_film'].append(np.mean(normalized_scores_p_film))
    global_stats['dist']['ld'].append(np.mean(normalized_scores_ld))

    # 3. Store Normalized Lengths (Mean of the successful steps for this iteration)
    global_stats['len']['without'].append(np.mean(normalized_lengths_without))
    global_stats['len']['l_film'].append(np.mean(normalized_lengths_l_film))
    global_stats['len']['p_film'].append(np.mean(normalized_lengths_p_film))
    global_stats['len']['ld'].append(np.mean(normalized_lengths_ld))

    with torch.no_grad():
        torch.cuda.empty_cache()

print("\n" + "="*50)
print("FINAL AGGREGATED RESULTS (Across 6 Goal Configs)")
print("="*50)

metrics = ['without', 'l_film', 'p_film', 'ld']
metric_names = ['SANS cond', 'l_FiLM', 'p_FiLM', 'Legibility Diff']

print("\n" + "="*80)
print("SPLIT ANALYSIS: Configs 1-3 vs Configs 4-6")
print("="*80)

# Helper function to print rows (optional, but keeps code clean)
def print_metric_group(title, stat_key):
    print(f"\n--- {title} ---")
    # Header row
    print(f"{'Method':<20} | {'Configs 1-3 (Mean +/- Std)':<28} | {'Configs 4-6 (Mean +/- Std)':<28}")
    print("-" * 82)
    
    for key, name in zip(metrics, metric_names):
        data = global_stats[stat_key][key]
        
        # Slice the data
        first_set = data[:3]  # Configs 1, 2, 3
        last_set = data[3:]   # Configs 4, 5, 6
        
        # Compute Stats for First Set
        m1 = np.mean(first_set)
        s1 = np.std(first_set)
        
        # Compute Stats for Last Set
        m2 = np.mean(last_set)
        s2 = np.std(last_set)
        
        print(f"{name:<20} | {m1:.4f} +/- {s1:.4f}{' '*13} | {m2:.4f} +/- {s2:.4f}")

# 1. Final Success Rates
print_metric_group("Success Rates", 'success')

# 2. Final Distances
print_metric_group("Normalized Distances to negative goal", 'dist')

# 3. Final Lengths
# print_metric_group("Normalized Lengths", 'len')
print_metric_group("Normalized Efficiencies", 'len')
