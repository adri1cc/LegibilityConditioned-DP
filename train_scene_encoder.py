# Train the scene encoder, with an associated decoder, for scenes with 2 objects

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import time

from utils.architecture_utils import SceneEncoder, SceneDecoder
from utils.dataset_utils import RandomSceneDataset

device = torch.device('cuda')

def train_scene_autoencoder(encoder, decoder, dataloader, optimizer, device, num_epochs=50):
    encoder.train()
    decoder.train()
    loss_fn = nn.MSELoss()
    loss_per_epoch = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            nobs = batch['obs'].to(device).float()
            goal_pos = nobs[:, -1, 7:10]
            object_pos = nobs[:, -1, 10:13].unsqueeze(1)

            context = encoder(goal_pos, object_pos)
            x_recon = decoder(context)

            # recompute target enriched scene (same as in encoder)
            B, N, _ = object_pos.shape
            rel_vectors = object_pos - goal_pos.unsqueeze(1)
            dists = torch.norm(rel_vectors, dim=-1, keepdim=True)
            enriched_objects = torch.cat([object_pos, rel_vectors, dists], dim=-1)
            goal_enriched = torch.cat([goal_pos, torch.zeros_like(goal_pos), torch.zeros(B, 1, device=device)], dim=-1).unsqueeze(1)
            target = torch.cat([goal_enriched, enriched_objects], dim=1)

            loss = loss_fn(x_recon, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        loss_per_epoch.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), loss_per_epoch, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()

    return encoder.state_dict()

encoder = SceneEncoder(
    obj_dim=7, num_objects=1
    ).to(device)

decoder = SceneDecoder(
    obj_dim=7, num_objects=1
).to(device)

dataset = RandomSceneDataset(size=20000)

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

optimizer = torch.optim.AdamW(
    params=list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-5, weight_decay=1e-6)

num_epochs = 50

encoder_state_dict = train_scene_autoencoder(encoder, decoder, dataloader, optimizer, device, num_epochs)
torch.save(encoder_state_dict, f"output/scene_encoder/se_{num_epochs}e_{time.strftime('%Y-%m-%d_%H-%M')}.pth")

with torch.no_grad():
    torch.cuda.empty_cache()