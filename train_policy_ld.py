# Train base, goal-conditioned Diffusion Policy

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os
from utils.dataset_utils import FrankaStateDataset
from utils.architecture_utils import ControlConditionalUnet1D

for i in range(1, 7):

    ## DATASET ##

    dataset_paths = [
        f"data/f_200_trajectories_{i}.hdf5"
        # "data/bezier_legible.hdf5", # z fixé à 0.1
        # "data/franka/hdf5/new_bezier_15000.hdf5", # z variable
        # "data/franka/hdf5/bezier_10000_leg2.hdf5" # Legible data
    ]
    # dataset_path = "data/franka/hdf5/dataset_franka_legibility_4.hdf5"
    for dataset_path in dataset_paths:
        if not os.path.isfile(dataset_path):
            print("No dataset found.")

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = FrankaStateDataset(
        dataset_paths=dataset_paths,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape", batch['action'].shape)

    ## UNET ##

    # Observation and action dimensions corrsponding to the Franka
    obs_dim = 10 # State (7 joints) + goal coordinates
    action_dim = 7 # Next state

    # Create network object
    noise_pred_net = ControlConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
    )

    # DDPMScheduler with 100 diffusion iterations
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

    # Device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)

    ## TRAINING ##

    num_epochs = 100
    goal_dim = 3 
    p_uncond = 0.1

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device).float()
                    naction = nbatch['action'].to(device).float()
                    # print(nobs.shape) # torch.Size([256, 2, 10])
                    # print(nclass.shape) # torch.Size([256])
                    B = nobs.shape[0]

                    nobs = nobs[:, :, :10]
                    
                    if torch.any(torch.isnan(nobs)) or torch.any(torch.isnan(naction)):
                        print("NaN detected in inputs!")

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]

                    # Supposons que les dernières dimensions correspondent au goal
                    obs_only = obs_cond[:, :, :-goal_dim]  # ex: état du robot
                    goal_only = obs_cond[:, :, -goal_dim:]  # ex: position du goal

                    # Masquage stochastique du goal avec p_uncond
                    B = obs_cond.shape[0]
                    mask = torch.rand(B, device=device) < p_uncond  # booléen : True → goal neutre

                    # Définir le goal neutre
                    goal_neutral = torch.zeros_like(goal_only)

                    # Remplacer le goal par goal neutre si mask=True
                    goal_cond = torch.where(mask.view(-1, 1, 1), goal_neutral, goal_only)

                    # Recomposer obs_cond = obs || goal
                    obs_cond = torch.cat([obs_only, goal_cond], dim=-1)

                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)
                    
                    if torch.any(torch.isnan(noisy_actions)) or torch.any(torch.isnan(noise_pred)):
                        print("NaN detected in intermediate outputs!")

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())

    # Saving checkpoint
    checkpoint_path = f"output/ckpt/f_legidiff_{i}.ckpt"

    checkpoint = {
        'model_state_dict': ema_noise_pred_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_idx,
        'loss': np.mean(epoch_loss),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    with torch.no_grad():
        torch.cuda.empty_cache()