# Script to evaluate legibility of Diffusion Policy with and without the legibility module

import os
import torch
import numpy as np
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils.architecture_utils import ControlConditionalUnet1D, SceneEncoder, FiLMGenerator
from utils.dataset_utils import normalize_data, unnormalize_data
from tqdm import tqdm
import sys
import pickle
import torch.nn as nn
import panda_py
import pandas as pd
import time
import random

from utils.inference_utils import rollout_to_goal
from utils.inference_utils_ld import rollout_to_goal_legdiff

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device
device = torch.device('cuda')

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# observation and action dimensions corrsponding to the output of Franka Gazebo
obs_dim = 10 # 10 or 16
obs_dim_legdiff = 10
action_dim = 7

# create network object
noise_pred_net = ControlConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
)

noise_pred_net_legdiff = ControlConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim_legdiff*obs_horizon
)

# Loading Pretrained Checkpoint
# ckpt_path = "ckpt/goal_dp_bezier.ckpt"
ckpt_path = "output/ckpt/f_dp_3.ckpt"
if not os.path.isfile(ckpt_path):
    exit("Checkpoint file not found.")
    
state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = state_dict["model_state_dict"]
ema_noise_pred_net = noise_pred_net
ema_noise_pred_net = ema_noise_pred_net.to(device)
ema_noise_pred_net.load_state_dict(state_dict, strict=False)

# ckpt_path_legdiff = "ckpt/legdiff_bezier_legible.ckpt"
# if not os.path.isfile(ckpt_path):
#     exit("Checkpoint file not found.")
ckpt_path_legdiff = "output/ckpt/f_legidiff_3.ckpt"
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

# l_film_generator = FiLMGenerator(
#     context_dim=64, feature_dims={1: 256, 2: 512, 3: 1024, 4: 1024, 5: 1024, 6:512, 7:256}
# ).to(device)

l_film_generator = FiLMGenerator(
    context_dim=64, feature_dims={4: 1024}
).to(device)

p_film_generator = FiLMGenerator(
    context_dim=64, feature_dims={4: 1024}
).to(device)


encoder.load_state_dict(torch.load('output/scene_encoder/se_20000_50e.pth'))

l_film_generator.load_state_dict(torch.load('output/mlp/f_legibility_2000e_3.pt'))
# l_film_generator.load_state_dict(torch.load('output/mlp/f_legibility_2000e_3_multi.pt'))

p_film_generator.load_state_dict(torch.load('output/mlp/f_predictability_2000e_3.pt'))
# film_generator.load_state_dict(torch.load('output/mlp/legibility_2000e_40_2_1.pt'))

ema_noise_pred_net.eval()
ema_noise_pred_net_legdiff.eval()
l_film_generator.eval()
p_film_generator.eval()
encoder.eval()

# Inference
# limit enviornment interaction to max_steps steps before termination
max_steps = 200

initial_joint = [0.00015335381064485176, -0.7855436163999139, -8.498953462776626e-05, -2.3561257643176736, -6.181224792900508e-05, 1.5713097735372683, 0.7854071770339326]  # position de départ fixe

def simulate_trajectory(actions, initial_joint):
    joint_positions = [initial_joint]
    for a in actions:
        joint_positions.append(a)  # ou joint_positions[-1] + a pour des actions relatives
    return np.stack(joint_positions)

def compute_distance_to_goal2(trajectory, goal_2):
    positions = trajectory
    distances = np.linalg.norm(positions - goal_2, axis=1)
    
    times = np.arange(1, len(trajectory) + 1)

    weighted_distances = distances / times

    return np.sum(weighted_distances)# /len(trajectory)

def compute_trajectory_length(trajectory):
    length = 0
    last_waypoint = None
    for waypoint in trajectory:
        if last_waypoint is not None:
            dist = np.linalg.norm(waypoint - last_waypoint)
            length = length + dist

        last_waypoint = waypoint

    return length

# with open("stats/bezier_state2.pkl", "rb") as f:
with open("stats/f_stats_3.pkl", "rb") as f:
    stats = pickle.load(f)
with open("stats/f_stats_3_ld.pkl", "rb") as f:
    stats_ld = pickle.load(f)

dists_with_half_latent = []
dists_multi_film = []
dists_multi_film2 = []
dists_multi_film3 = []
dists_film = []
dists_without_latent = []
dists_ld = []
dists_l_film = []
dists_p_film = []

half_dists_with_latent = []
half_dists_without_latent = []
half_dists_leg_diff = []
half_dists_with_half = []

legibilities_with_latent = []
legibilities_without_latent = []
legibilities_leg_diff = []

successes_multi_film = []
successes_film = []
successes_l_film = []
successes_p_film = []
successes_without_latent = []
successes_leg_diff = []

legibilities_with_latent_b = []
legibilities_without_latent_b = []
legibilities_leg_diff_b = []

inference_times_with_half_latent = []
inference_times_with_latent = []
inference_times_without_latent = []
inference_times_leg_diff = []

lengths_ld = []
lengths_film = []
lengths_l_film = []
lengths_p_film = []
lengths_multi_film = []
lengths_multi_film2 = []
lengths_multi_film3 = []
lengths_without_latent = []

len_traj_film = []
len_traj_l_film = []
len_traj_p_film = []
len_traj_multi_film = []
len_traj_without_latent = []
len_traj_leg_diff = []
len_traj_multi_film2 = []
len_traj_multi_film3 = []

trajectories_without = []
trajectories_film = []
trajectories_l_film = []
trajectories_p_film = []
trajectories_ld = []

with open("stats/f_legibility_bounds_3.txt", "r") as f:
    smallest_len_s = int(f.readline().strip())
    min_dist = float(f.readline().strip())
    max_dist = float(f.readline().strip())
# print(smallest_len_s)

with open("stats/f_length_bounds_3.txt") as f:
    min_length = float(f.readline().strip())
    max_length = float(f.readline().strip())

for i in range(100):
    # print(f"Evaluation {i}")
    if random.uniform(0, 1) < 0.5:
        goal_1 = np.array([0.5, -0.1, 0.1])
        goal_2 = np.array([0.65, 0.3, 0.1])
        # goal_2 = np.array([0.5, 0.1, 0.1])
    else:
        # goal_1 = np.array([0.5, 0.1, 0.1])
        goal_1 = np.array([0.65, 0.3, 0.1])
        goal_2 = np.array([0.5, -0.1, 0.1])

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
    # successes_film.append(success_film)
    successes_without_latent.append(success_without)
    successes_leg_diff.append(success_ld)

    if (success_without == True) and (success_ld == True) and (success_l_film == True) and (success_p_film == True):
        smallest_len = smallest_len_s
        for j in [len(traj_ld), len(traj_without), len(traj_l_film), len(traj_p_film)]:#, len(traj_with_half)]:
            if j < smallest_len:
                smallest_len = j
                # print(smallest_len)

        for traj in [traj_ld, traj_without, traj_l_film, traj_p_film]: #, traj_with_half]:
            if len(traj) > smallest_len:
                indices = np.linspace(0, len(traj) - 1, smallest_len, dtype=int)
                traj_sampled = [traj[i] for i in indices]
                if traj is traj_ld:
                    # print("sampling LD traj")
                    traj_ld = traj_sampled
                elif traj is traj_l_film:
                    # print("sampling FiLM traj")
                    traj_l_film = traj_sampled
                elif traj is traj_p_film:
                    traj_p_film = traj_sampled
                elif traj is traj_without:
                    # print("sampling DP traj")
                    traj_without = traj_sampled

        len_traj_p_film.append(len(traj_p_film))
        len_traj_l_film.append(len(traj_l_film))
        len_traj_without_latent.append(len(traj_without))
        len_traj_leg_diff.append(len(traj_ld))

        # dist_film = compute_distance_to_goal2(traj_film, goal_2)
        dist_l_film = compute_distance_to_goal2(traj_l_film, goal_2)
        dist_p_film = compute_distance_to_goal2(traj_p_film, goal_2)
        dist_without = compute_distance_to_goal2(traj_without, goal_2)
        dist_ld = compute_distance_to_goal2(traj_ld, goal_2)

        # length_film = compute_trajectory_length(traj_film)
        length_l_film = compute_trajectory_length(traj_l_film)
        length_p_film = compute_trajectory_length(traj_p_film)
        length_without = compute_trajectory_length(traj_without)
        length_ld = compute_trajectory_length(traj_ld)

        lengths_ld.append(length_ld)
        # lengths_film.append(length_film)
        lengths_l_film.append(length_l_film)
        lengths_p_film.append(length_p_film)
        lengths_without_latent.append(length_without)
        
        # dists_film.append(dist_film)
        dists_l_film.append(dist_l_film)
        dists_p_film.append(dist_p_film)
        dists_without_latent.append(dist_without)
        dists_ld.append(dist_ld)

        # trajectories_film.append(traj_film)
        trajectories_l_film.append(traj_l_film)
        trajectories_p_film.append(traj_p_film)
        trajectories_ld.append(traj_ld)
        trajectories_without.append(traj_without)

        # print("Success.")
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
# normalized_scores_film = [(d - min_dist) / (max_dist - min_dist) for d in dists_film]
normalized_scores_l_film = [(d - min_dist) / (max_dist - min_dist) for d in dists_l_film]
normalized_scores_p_film = [(d - min_dist) / (max_dist - min_dist) for d in dists_p_film]
normalized_scores_without = [(d - min_dist) / (max_dist - min_dist) for d in dists_without_latent]

normalized_lengths_ld = [(l - min_length) / (max_length - min_length) for l in lengths_ld]
# normalized_lengths_film = [(l - min_length) / (max_length - min_length) for l in lengths_film]
normalized_lengths_l_film = [(l - min_length) / (max_length - min_length) for l in lengths_l_film]
normalized_lengths_p_film = [(l - min_length) / (max_length - min_length) for l in lengths_p_film]
normalized_lengths_without = [(l - min_length) / (max_length - min_length) for l in lengths_without_latent]

print("Success rate SANS conditionnement : ", np.mean(successes_without_latent))
# print("Success rate AVEC conditionnement (FiLM) : ", np.mean(successes_film))
print("Success rate AVEC conditionnement (l_FiLM) : ", np.mean(successes_l_film))
print("Success rate AVEC conditionnement (p_FiLM) : ", np.mean(successes_p_film))
# print("Success rate AVEC 0.5*vecteur latent: ", np.mean(successes_with_half_latent))
print("Success rate Legibility Diffuser: ", np.mean(success_ld))

print("Moyenne distance à goal_2 SANS conditionnement :", np.mean(normalized_scores_without), "+/-", np.std(normalized_scores_without))
# print("Moyenne distance à goal_2 AVEC conditionnement (FiLM) :", np.mean(normalized_scores_film), "+/-", np.std(normalized_scores_film))
print("Moyenne distance à goal_2 AVEC conditionnement (l_FiLM) :", np.mean(normalized_scores_l_film), "+/-", np.std(normalized_scores_l_film))
print("Moyenne distance à goal_2 AVEC conditionnement (p_FiLM) :", np.mean(normalized_scores_p_film), "+/-", np.std(normalized_scores_p_film))
print("Moyenne distance à goal_2 Legibility Diffuser :", np.mean(normalized_scores_ld), "+/-", np.std(normalized_scores_ld))

print("Moyenne longueur trajectoire SANS conditionnement :", np.mean(normalized_lengths_without), "+/-", np.std(normalized_lengths_without))
# print("Moyenne longueur trajectoire AVEC conditionnement (FiLM) :", np.mean(normalized_lengths_film), "+/-", np.std(normalized_lengths_film))
print("Moyenne longueur trajectoire AVEC conditionnement (l_FiLM) :", np.mean(normalized_lengths_l_film), "+/-", np.std(normalized_lengths_l_film))
print("Moyenne longueur trajectoire AVEC conditionnement (p_FiLM) :", np.mean(normalized_lengths_p_film), "+/-", np.std(normalized_lengths_p_film))
print("Moyenne longueur trajectoire Legibility Diffuser :", np.mean(normalized_lengths_ld), "+/-", np.std(normalized_lengths_ld))

# print("Nombre steps moyen SANS vecteur latent :", np.mean(len_traj_without_latent))
# print("Nombre steps moyen AVEC vecteur latent (FiLM) :", np.mean(len_traj_film))
# print("Nombre steps moyen AVEC vecteur latent (Addition) :", np.mean(len_traj_multi_film))
# print("Nombre steps moyen Legibility Diffuser :", np.mean(len_traj_leg_diff))

with torch.no_grad():
    torch.cuda.empty_cache()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # --- Visualisation des trajectoires ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Vérifier qu’on a au moins une trajectoire valide
# if len(traj_l_film) > 0 and len(traj_p_film) > 0 and len(traj_without) > 0 and len(traj_ld) > 0:
#     # Convertir en arrays numpy pour indexer facilement

#     # Tracer les trajectoires
#     for traj in trajectories_without:
#         traj_without = np.array(traj)
#         ax.plot(
#             traj_without[:, 0], traj_without[:, 1], traj_without[:, 2],
#             color='tab:orange', linewidth=2
#         )
#     for traj in trajectories_l_film:
#         traj_l_film = np.array(traj)
#         ax.plot(
#             traj_l_film[:, 0], traj_l_film[:, 1], traj_l_film[:, 2],
#             color='green', linewidth=2
#         )
#     for traj in trajectories_p_film:
#         traj_p_film = np.array(traj)
#         ax.plot(
#             traj_p_film[:, 0], traj_p_film[:, 1], traj_p_film[:, 2],
#             color='red', linewidth=2
#         )
#     for traj in trajectories_ld:
#         traj_ld = np.array(traj)
#         ax.plot(
#             traj_ld[:, 0], traj_ld[:, 1], traj_ld[:, 2],
#             color='blue', linewidth=2
#         )

#     # Points de départ et d’arrivée
#     ax.scatter(goal_1[0], goal_1[1], goal_1[2],
#                color='orange', marker='*', s=100, label='Goal 1')
#     ax.scatter(goal_2[0], goal_2[1], goal_2[2],
#                color='purple', marker='*', s=100, label='Goal 2')

#     ax.set_title("Visualisation 3D des trajectoires générées", fontsize=14)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend()
#     ax.grid(True)

#     plt.tight_layout()
#     plt.show()