# Train legiility MLP, with a pre-trained Diffusion Policy model and a pre-trained scene encoder

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numpy as np
import os
import pickle
import time

from utils.architecture_utils import ControlConditionalUnet1D, FiLMGenerator, SceneEncoder
from utils.dataset_utils import FrankaStateDataset

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
obs_dim = 10
action_dim = 7
class_dim = 1

# legible demonstration data
dataset_path = ["data/5000_legible.hdf5",]

# Use the stats file created from train_policy.py
with open("stats/stats_13000.pkl", "rb") as f:
    stats = pickle.load(f)

# create dataset from file
dataset = FrankaStateDataset(
    dataset_paths=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    stats=stats
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    drop_last = True
)

device = torch.device('cuda')

# UNet
noise_pred_net = ControlConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
).to('cuda')

# loading Pretrained Checkpoint
ckpt_path = "output/ckpt/dp_25000.ckpt" 
if not os.path.isfile(ckpt_path):
    exit("Checkpoint file not found.")

state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = state_dict["model_state_dict"]
ema_noise_pred_net = noise_pred_net
ema_noise_pred_net = ema_noise_pred_net.to(device)
ema_noise_pred_net.load_state_dict(state_dict, strict=False)

# freezing the model
ema_noise_pred_net = ema_noise_pred_net.eval()
for p in ema_noise_pred_net.parameters():
    p.requires_grad = False

num_epochs = 100

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

epoch_losses = []
latent_history = []
grad_step_count = 0

# loading scene encoder
encoder = SceneEncoder(
    obj_dim=7, num_objects=1
).to(device)
encoder.load_state_dict(torch.load('output/scene_encoder/se_20000_50e.pth'))
encoder.eval()

# legibility MLP
film_generator = FiLMGenerator(
    context_dim=64, feature_dims={1: 256, 2: 512, 3: 1024, 4: 1024, 5: 1024, 6:512, 7:256}
).to(device)

optimizer = torch.optim.AdamW(
    params=list(film_generator.parameters()),
    lr=1e-5, weight_decay=1e-6)

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

                # goal to reach
                goal_pos = nobs[:, -1, 7:10]
                # negative goal
                object_pos = nobs[:, -1, 10:13].unsqueeze(1)

                nobs = nobs[:, :, :10]

                if torch.any(torch.isnan(nobs)) or torch.any(torch.isnan(naction)):
                    print("NaN detected in inputs!")

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                context_vector = encoder(goal_pos, object_pos)

                # generating a second FiLM conditioning terms depending on the context
                modulations = film_generator(context_vector)
                control_cond = []
                for i in range(1, 8):
                    gamma, beta = modulations[i]
                    control_cond.append([gamma.unsqueeze(-1), beta.unsqueeze(-1)])

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
                noise_pred = ema_noise_pred_net(
                    noisy_actions, 
                    timesteps,
                    global_cond=obs_cond, 
                    control_cond=control_cond,
                    method="film"
                )
                
                if torch.any(torch.isnan(noisy_actions)) or torch.any(torch.isnan(noise_pred)):
                    print("NaN detected in intermediate outputs!")

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                grad_step_count += 1
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        epoch_losses.append(np.mean(epoch_loss))

        tglobal.set_postfix(loss=np.mean(epoch_loss))

torch.save(film_generator.state_dict(), f"output/mlp/legibility_{num_epochs}e_{time.strftime('%Y-%m-%d_%H-%M')}.pt")

with torch.no_grad():
    torch.cuda.empty_cache()

print(f"Total gradient steps: {grad_step_count}")

plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), epoch_losses, label='Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss per Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("loss_curve.png") 
plt.show()


# To avoid log(0)
epsilon = 1e-8
log_epoch_losses = [np.log(loss + epsilon) for loss in epoch_losses]

plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), log_epoch_losses, label='log(Loss)', color='orange')
plt.xlabel("Epoch")
plt.ylabel("log(MSE Loss)")
plt.title("log(Loss) per Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("log_loss_curve.png")
plt.show()