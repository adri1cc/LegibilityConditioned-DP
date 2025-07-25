import torch
import numpy as np
import collections
from utils.dataset_utils import normalize_data, unnormalize_data
import panda_py

def random_main_goal():
    goal = np.array([     
        np.random.uniform(0.35, 0.65), # x
        np.random.uniform(-0.2, 0.2),  # y
        0.1                            # z
    ])
    return goal.tolist()

def random_goal(exclude, min_dist=0.1, max_dist=0.8):
    while True:
        goal = np.array([
            np.random.uniform(0.3, 0.7),  # x
            np.random.uniform(-0.3, 0.3), # y
            0.1                           # z
        ])
        if (np.linalg.norm(goal - np.array(exclude)) >= min_dist) and (np.linalg.norm(goal - np.array(exclude)) <= max_dist):
            return goal.tolist()

def rollout_to_goal(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, 
        encoder=None, film=None, method=None, max_steps=200, num_diffusion_iters=100, device=torch.device('cuda')
        ):

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    action_dim = 7

    goal_tensor = torch.tensor([goal_1], dtype=torch.float32, device=device)
    obj_tensor = torch.tensor([[goal_2]], dtype=torch.float32, device=device)

    if film is not None and encoder is not None:

        if method=="film":

            context_vector = encoder(goal_tensor, obj_tensor)
            modulations = film(context_vector)
            control_cond = []
            for i in range(1, 8):
                gamma, beta = modulations[i]
                control_cond.append([gamma.unsqueeze(-1), beta.unsqueeze(-1)])

    else:
        control_cond = None

    success = False

    obs = np.concatenate((initial_joint, goal_1), axis=0)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    current_joint = initial_joint.copy()
    robot_pose = np.asanyarray(panda_py.fk(current_joint)[:3, 3])

    trajectory = [robot_pose.copy()]
    joint_trajectory = [current_joint.copy()]

    for step in range(max_steps):
        obs_seq = np.stack(obs_deque)
        nobs_robot_state = normalize_data(obs_seq[:, :7], stats=stats['robot_state'])
        nobs = np.concatenate([nobs_robot_state, obs_seq[:, 7:]], axis=-1)
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        with torch.no_grad():
            naction = torch.randn((1, pred_horizon, action_dim), device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters)

            if step*8<15:
                for k in noise_scheduler.timesteps:
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                        control_cond=control_cond,
                        method=method
                    )
                    naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            else:
                for k in noise_scheduler.timesteps:
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                    )
                    naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            action_pred = naction.detach().cpu().numpy()[0]
            action_pred = unnormalize_data(action_pred, stats=stats['action'])

        for a in action_pred[:action_horizon]:
            current_joint = a 
            obs = np.concatenate((current_joint, goal_1), axis=0)
            obs_deque.append(obs)
            joint_trajectory.append(current_joint.copy())
            robot_pose = np.asanyarray(panda_py.fk(current_joint)[:3, 3])
            trajectory.append(robot_pose)

            dist_to_goal = np.linalg.norm(robot_pose - goal_1[:3])
            # Success criteria for evaluation
            if dist_to_goal < 0.035:

                success = True
                return np.stack(trajectory), success

    return np.stack(trajectory), success