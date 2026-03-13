import torch
import numpy as np
import collections
from utils.dataset_utils import normalize_data, unnormalize_data
import panda_py

def random_main_goal():
    goal = np.array([     
        np.random.uniform(0.3, 0.7),  # x
        np.random.uniform(-0.3, 0.3), # y
        0.1                           # z
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

def rollout_to_goal_legdiff(
        initial_joint, goal_1, goal_2, ema_noise_pred_net, noise_scheduler, stats, 
        max_steps=20, num_diffusion_iters=100, device=torch.device('cuda')
        ):
    
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    action_dim = 7

    success = False
    
    goal_3 = np.array([0.0, 0.0, 0.0])

    obs_1 = np.concatenate((initial_joint, goal_1), axis=0)
    obs_2 = np.concatenate((initial_joint, goal_2), axis=0)
    obs_null = np.concatenate((initial_joint, goal_3), axis=0)

    obs_1_deque = collections.deque(
        [obs_1] * obs_horizon, maxlen=obs_horizon)
    obs_2_deque = collections.deque(
        [obs_2] * obs_horizon, maxlen=obs_horizon)
    obs_null_deque = collections.deque(
        [obs_null] * obs_horizon, maxlen=obs_horizon)
    current_joint = initial_joint.copy()
    robot_pose = np.asanyarray(panda_py.fk(current_joint)[:3, 3])

    trajectory = [robot_pose.copy()]
    joint_trajectory = [current_joint.copy()]

    # alpha = 0.9
    # gamma = 0.98
    # w_t = 15.0

    alpha = 0.9
    gamma = 0.5
    w_t = 4.5

    for step in range(max_steps):

        obs_seq_1 = np.stack(obs_1_deque)
        nobs_robot_state_1 = normalize_data(obs_seq_1[:, :7], stats=stats['robot_state'])
        nobs_1 = np.concatenate([nobs_robot_state_1, obs_seq_1[:, 7:]], axis=-1)
        # nobs_1 = normalize_data(obs_seq_1, stats=stats_legdiff['obs'])
        nobs_1 = torch.from_numpy(nobs_1).to(device, dtype=torch.float32)

        obs_seq_2 = np.stack(obs_2_deque)
        nobs_robot_state_2 = normalize_data(obs_seq_2[:, :7], stats=stats['robot_state'])
        nobs_2 = np.concatenate([nobs_robot_state_2, obs_seq_2[:, 7:]], axis=-1)
        # nobs_2 = normalize_data(obs_seq_2, stats=stats_legdiff['obs'])
        nobs_2 = torch.from_numpy(nobs_2).to(device, dtype=torch.float32)

        obs_seq_null = np.stack(obs_null_deque)
        nobs_robot_state_null = normalize_data(obs_seq_null[:, :7], stats=stats['robot_state'])
        nobs_null = np.concatenate([nobs_robot_state_null, obs_seq_null[:, 7:]], axis=-1)
        # nobs_null = normalize_data(obs_seq_null, stats=stats_legdiff['obs'])
        nobs_null = torch.from_numpy(nobs_null).to(device, dtype=torch.float32)

        obs_cond_1 = nobs_1.unsqueeze(0).flatten(start_dim=1).to(device)
        obs_cond_2 = nobs_2.unsqueeze(0).flatten(start_dim=1).to(device)
        obs_cond_null = nobs_null.unsqueeze(0).flatten(start_dim=1).to(device)

        # init action avec bruit
        with torch.no_grad():
            naction = torch.randn((1, pred_horizon, action_dim), device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                noise_pred_goal = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond_1,
                )

                noise_pred_anti = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond_2,
                )

                noise_pred_null = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond_null,
                )

                noise_pred_guided = (
                    (1 + w_t) * noise_pred_goal
                    - alpha * w_t * noise_pred_anti
                    - (1 - alpha) * w_t * noise_pred_null
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred_guided,
                    timestep=k,
                    sample=naction
                ).prev_sample

            w_t *= gamma

            action_pred = naction.detach().cpu().numpy()[0]
            action_pred = unnormalize_data(action_pred, stats=stats['action'])

        # appliquer les premières actions
        for a in action_pred[:action_horizon]:
            current_joint = a  # ou current_joint + a si actions relatives

            obs_1 = np.concatenate((current_joint, goal_1), axis=0)
            obs_2 = np.concatenate((current_joint, goal_2), axis=0)
            obs_null = np.concatenate((current_joint, goal_3), axis=0)

            obs_1_deque.append(obs_1)
            obs_2_deque.append(obs_2)
            obs_null_deque.append(obs_null)

            joint_trajectory.append(current_joint.copy())
            robot_pose = np.asanyarray(panda_py.fk(current_joint)[:3, 3])
            trajectory.append(robot_pose)
            # critère d'arrêt
            dist_to_goal = np.linalg.norm(robot_pose - goal_1[:3])
            if dist_to_goal < 0.035:
                # print(robot_pose)
                success = True
                return np.stack(trajectory), success

    return np.stack(trajectory), success

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

            gamma1, beta1 = modulations[1]
            gamma2, beta2 = modulations[2]
            gamma3, beta3 = modulations[3]

            gamma4, beta4 = modulations[4]
            gamma5, beta5 = modulations[5]
            gamma6, beta6 = modulations[6]
            gamma7, beta7 = modulations[7]

            gamma1 = gamma1.unsqueeze(-1)
            beta1 = beta1.unsqueeze(-1)
            gamma2 = gamma2.unsqueeze(-1)
            beta2 = beta2.unsqueeze(-1)
            gamma3 = gamma3.unsqueeze(-1)
            beta3 = beta3.unsqueeze(-1)

            gamma4 = gamma4.unsqueeze(-1)
            beta4 = beta4.unsqueeze(-1)
            gamma5 = gamma5.unsqueeze(-1)
            beta5 = beta5.unsqueeze(-1)
            gamma6 = gamma6.unsqueeze(-1)
            beta6 = beta6.unsqueeze(-1)
            gamma7 = gamma7.unsqueeze(-1)
            beta7 = beta7.unsqueeze(-1)

            control_cond = [[gamma1, beta1], [gamma2, beta2], [gamma3, beta3], [gamma4, beta4], [gamma5, beta5], [gamma6, beta6], [gamma7, beta7]]

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