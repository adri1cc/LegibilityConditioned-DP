# The different architecture alements used for this work
# Most classes follow the original Diffusion Policy implementation

from typing import Union
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond, control_cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]
            control_cond : Optional[Tuple[Tensor, Tensor]] - A tuple (gamma_ctrl, beta_ctrl)

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        # Style Conditioning
        if control_cond is not None:
            gamma_ctrl, beta_ctrl = control_cond
            out = gamma_ctrl * out + beta_ctrl    

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class ControlConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines number of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim 

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        x_channels = down_dims[-1]
        self.control_concat_proj = nn.Conv1d(x_channels * 2, x_channels, kernel_size=1)

        control_channels = 1024
        self.control_gate_proj = nn.Conv1d(control_channels, x_channels, kernel_size= 1)
        self.control_film_proj = nn.Conv1d(control_channels, x_channels * 2, kernel_size=1)

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )


    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None,
            control_cond=None,
            method=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        control_cond: Optional[List[Tensor]] - List of tensors used for control conditioning at each level of the UNet.
        method: string

        Returns: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # Condition globale
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        layer = 0
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            cond = control_cond[layer] if control_cond is not None else None
            x = resnet(x, global_feature, control_cond=cond)
            x = resnet2(x, global_feature, control_cond=cond)
            h.append(x)
            x = downsample(x)
            layer +=1

        for mid_module in self.mid_modules:
            cond = control_cond[layer] if control_cond is not None else None
            x = mid_module(x, global_feature, control_cond=cond)
            layer +=1

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            cond = control_cond[layer] if control_cond is not None else None
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature, control_cond=cond)
            x = resnet2(x, global_feature, control_cond=cond)
            x = upsample(x)
            layer +=1

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


class FiLMGenerator(nn.Module):
    def __init__(self, context_dim, feature_dims):
        """
        context_dim: int, dimension of the context vector
        feature_dims: dict with keys {1, 3, 5} and values = feature_dim for each block
        Example: {1: 64, 3: 128, 5: 64}
        """
        super().__init__()
        self.gamma_fcs = nn.ModuleDict()
        self.beta_fcs = nn.ModuleDict()

        for block_id, feat_dim in feature_dims.items():
            self.gamma_fcs[str(block_id)] = nn.Linear(context_dim, feat_dim)
            self.beta_fcs[str(block_id)] = nn.Linear(context_dim, feat_dim)

    def forward(self, context_vector):
        """
        context_vector: (B, context_dim)

        Returns: dict with keys {1, 3, 5}, each containing (gamma, beta) of shape (B, feature_dim)
        """
        modulations = {}
        for block_id in self.gamma_fcs.keys():
            gamma = self.gamma_fcs[block_id](context_vector)
            beta = self.beta_fcs[block_id](context_vector)
            modulations[int(block_id)] = (gamma, beta)
        return modulations
    
class SceneEncoder(nn.Module):
    def __init__(self, obj_dim=7, num_objects=1, hidden_dim=128, context_dim=64):
        """
        obj_dim (int): Dimensionality of each enriched object vector (default: 7).
        num_objects (int): Number of objects in the scene (excluding the goal).
        hidden_dim (int): Dimensionality of the hidden layers in the MLP.
        context_dim (int): Dimensionality of the output context vector.
        """
        super().__init__()
        input_dim = (num_objects + 1) * obj_dim  # goal + N objects
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )

    def forward(self, goal_pos, object_positions):
        """
        Inputs:
        goal_pos (Tensor): Goal position tensor of shape (B, 3), where B is the batch size.
        object_positions (Tensor): Object positions tensor of shape (B, N, 3), where N = num_objects.

        Returns:
        Tensor: Encoded context vector of shape (B, context_dim).
        """
        B, N, D = object_positions.shape
        rel_vectors = object_positions - goal_pos.unsqueeze(1)  # (B, N, 3)
        dists = torch.norm(rel_vectors, dim=-1, keepdim=True)   # (B, N, 1)
        enriched_objects = torch.cat([object_positions, rel_vectors, dists], dim=-1)  # (B, N, 7)

        zero_vec = torch.zeros_like(goal_pos)
        zero_dist = torch.zeros(B, 1, device=goal_pos.device)
        goal_enriched = torch.cat([goal_pos, zero_vec, zero_dist], dim=-1).unsqueeze(1)  # (B, 1, 7)

        all_objects = torch.cat([goal_enriched, enriched_objects], dim=1)  # (B, N+1, 7)
        x = all_objects.view(B, -1)  # (B, (N+1)*7)
        context = self.encoder(x)
        return context


class SceneDecoder(nn.Module):
    def __init__(self, obj_dim=7, num_objects=5, hidden_dim=128, context_dim=64):
        """
        obj_dim (int): Dimensionality of each enriched object vector (default: 7).
        num_objects (int): Number of objects in the scene (excluding the goal).
        hidden_dim (int): Dimensionality of the hidden layers in the MLP.
        context_dim (int): Dimensionality of the input context vector.
        """
        super().__init__()
        self.num_objects = num_objects
        output_dim = (num_objects + 1) * obj_dim
        self.decoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, context_vector):
        """
        Input:
        context_vector (Tensor): Context vector of shape (B, context_dim), representing the encoded scene.

        Returns:
        Tensor: Reconstructed enriched vectors of shape (B, N+1, obj_dim), where the first vector corresponds to the goal and the rest to the objects.
        """
        x_recon = self.decoder(context_vector)  # (B, (N+1)*7)
        return x_recon.view(context_vector.shape[0], self.num_objects + 1, -1)  # (B, N+1, 7)