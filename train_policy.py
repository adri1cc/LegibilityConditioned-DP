# Train base, goal-conditioned Diffusion Policy

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import os
import time

from utils.dataset_utils import FrankaStateDataset
from utils.architecture_utils import ControlConditionalUnet1D

## DATASET ##

# dataset_paths used to train goal_dp_bezier_25000.ckpt : 15000_nlegible.hdf5 and 10000_legible.hdf5
dataset_paths = [
    "data/8635_nlegible.hdf5",
    "data/5000_legible.hdf5"
]

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
# will create a normalization stats file if no stats file is given as an input
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

# observation and action dimensions corrsponding to the Franka
obs_dim = 10 # state (7 joints) + goal coordinates
action_dim = 7 # next state

# create network object
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

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

## TRAINING ##

num_epochs = 100

# Exponential Moving Average
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# standard ADAM optimizer
# note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# cosine LR schedule with linear warmup
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
                B = nobs.shape[0]

                # To only keep the robot state and the goal coordinates in nobs
                nobs = nobs[:, :, :10]
                
                if torch.any(torch.isnan(nobs)) or torch.any(torch.isnan(naction)):
                    print("NaN detected in inputs!")

                # observation for FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]

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

# weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

# Saving checkpoint
checkpoint_path = f"output/ckpt/diffusion_policy_{num_epochs}e_{time.strftime('%Y-%m-%d_%H-%M')}.ckpt" 

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