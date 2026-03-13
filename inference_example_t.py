# Visualize generated trajectories with Diffusion Policy, with or without using the legibility module, and Legibility Diffuser.

import os
import torch
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import pickle
import matplotlib.pyplot as plt

from utils.architecture_utils import ControlConditionalUnet1D, SceneEncoder, FiLMGenerator
from utils.inference_utils_t import random_main_goal, random_goal, rollout_to_goal, rollout_to_goal_legidiff

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

device = torch.device('cuda')

obs_horizon = 2
obs_dim = 5
action_dim = 3

noise_pred_net = ControlConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
)

noise_pred_net_ld = ControlConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
)

# load pretrained checkpoint
# ckpt_path = "output/ckpt/dp_25000.ckpt"
ckpt_path = "output/ckpt/t_dp_1.ckpt"
if not os.path.isfile(ckpt_path):
    exit("Checkpoint file not found.")
    
state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = state_dict["model_state_dict"]
ema_noise_pred_net = noise_pred_net
ema_noise_pred_net = ema_noise_pred_net.to(device)
ema_noise_pred_net.load_state_dict(state_dict, strict=False)
ema_noise_pred_net.eval()

ckpt_path_ld = "output/ckpt/t_legidiff_1.ckpt"
if not os.path.isfile(ckpt_path_ld):
    exit("Checkpoint file not found.")
state_dict = torch.load(ckpt_path_ld, map_location=device, weights_only=False)
state_dict = state_dict["model_state_dict"]
ema_noise_pred_net_ld = noise_pred_net_ld
ema_noise_pred_net_ld = ema_noise_pred_net_ld.to(device)
ema_noise_pred_net_ld.load_state_dict(state_dict, strict=False)
ema_noise_pred_net_ld.eval()

encoder = SceneEncoder(
    obj_dim=7, num_objects=1
).to(device)

film_generator = FiLMGenerator(
    context_dim=64, feature_dims={4: 1024}
).to(device)

# load scene encoder checkpoint 
encoder.load_state_dict(torch.load('output/scene_encoder/t_se_20000_50e.pth'))
encoder.eval()

# load legibility MLP
# film_generator.load_state_dict(torch.load('output/mlp/t_legibility_2000e_1.pt'))
film_generator.load_state_dict(torch.load('output/mlp/t_predictability_2000e_1.pt'))
film_generator.eval()

# initial Franka joints values
initial_joint = [0.0, 0.0, 0.0]  # position de départ fixe

def simulate_trajectory(actions, initial_joint):
    joint_positions = [initial_joint]
    for a in actions:
        joint_positions.append(a)
    return np.stack(joint_positions)

with open("stats/t_stats_1.pkl", "rb") as f:
    stats = pickle.load(f)
with open("stats/t_stats_1_ld.pkl", "rb") as f:
    stats_ld = pickle.load(f)

# choose preicse goal positions or randomize them
# note that in the training datasets z is fixed, with value 0.1
# goal_1 = random_main_goal()
# goal_2 = random_goal(goal_1)
goal_1 = np.array([2.0, 0.2])
goal_2 = np.array([2.0, -0.2])

traj_film, success_film = rollout_to_goal(
    initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, encoder=encoder, film=film_generator, method="film")

traj_without, success_without = rollout_to_goal(
    initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, method=None)

traj_ld, success_ld = rollout_to_goal_legidiff(
    initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats_ld)

print(success_without, success_film, success_ld)

if (success_without == True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(*goal_1, color='green', s=100, label='goal 1')
    ax.scatter(*goal_2, color='red', s=100, label='goal 2')
    
    if traj_film is not None:
        ax.plot(traj_film[:, 0], traj_film[:, 1], color='green', label='DP + Legibility')

    if traj_without is not None:
        ax.plot(traj_without[:, 0], traj_without[:, 1], color='orange', label='Diffusion Policy')

    if traj_ld is not None:
        ax.plot(traj_ld[:, 0], traj_ld[:, 1], color='blue', label='Legibility Diffuser')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Trajectories reaching goal 1")
    ax.legend()
    ax.grid(True)

    plt.show()

with torch.no_grad():
    torch.cuda.empty_cache()