""" This file contains nets used for Diffusion Policy. """
import math
from typing import Union

import torch
import torch.nn as nn
from robomimic.models.diffusion_policy_nets import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    ConditionalResidualBlock1D,
    ConditionalUnet1D
)


class DoubleConditionalUnet1D(ConditionalUnet1D):
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
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__(
            input_dim,
            global_cond_dim,
            diffusion_step_embed_dim,
            down_dims,
            kernel_size,
            n_groups
        )

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            nn.Linear(2, dsed),
            nn.Mish(),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        self.diffusion_step_encoder = diffusion_step_encoder

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            t: Union[torch.Tensor, float, int], 
            r: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        # timesteps = concat of t and r
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float, device=sample.device).view(-1, 1)
            r = torch.tensor([r], dtype=torch.float, device=sample.device).view(-1, 1) 
            timesteps = torch.cat([t, t-r], dim=1).view(-1, 2)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(sample.device)
            r = r[None].to(sample.device)
            timesteps = torch.cat([t, t-r], dim=1).view(-1, 2)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])
        else:
            timesteps = torch.cat([t, t-r], dim=1).view(-1, 2)

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
