# This model was ported from Google's mediapipe (Apache 2.0 license) to pytorch using Liam Schoneveld's github repository https://github.com/nlml/deconstruct-mediapipe

import torch
from torch import nn


class MLPMixerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        num_patches,
        hidden_units_mlp1,
        hidden_units_mlp2,
        dropout_rate=0.0,
        eps1=0.0000010132789611816406,
        eps2=0.0000010132789611816406,
    ):
        super().__init__()
        self.mlp_token_mixing = nn.Sequential(
            nn.Conv2d(num_patches, hidden_units_mlp1, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_units_mlp1, num_patches, 1),
        )
        self.mlp_channel_mixing = nn.Sequential(
            nn.Conv2d(in_dim, hidden_units_mlp2, 1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_units_mlp2, in_dim, 1),
        )
        self.norm1 = nn.LayerNorm(in_dim, bias=False, elementwise_affine=True, eps=eps1)
        self.norm2 = nn.LayerNorm(in_dim, bias=False, elementwise_affine=True, eps=eps2)

    def forward(self, x):
        x_1 = self.norm1(x)
        mlp1_outputs = self.mlp_token_mixing(x_1)
        x = x + mlp1_outputs
        x_2 = self.norm2(x)
        mlp2_outputs = self.mlp_channel_mixing(x_2.permute(0, 3, 2, 1))
        x = x + mlp2_outputs.permute(0, 3, 2, 1)
        return x


class MediaPipeBlendshapesMLPMixer(nn.Module):
    def __init__(
        self,
        in_dim=64,
        num_patches=97,
        hidden_units_mlp1=384,
        hidden_units_mlp2=256,
        num_blocks=4,
        dropout_rate=0.0,
        output_dim=52,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(146, 96, kernel_size=1)
        self.conv2 = nn.Conv2d(2, 64, kernel_size=1)
        self.extra_token = nn.Parameter(torch.randn(1, 64, 1, 1), requires_grad=True)
        self.mlpmixer_blocks = nn.Sequential(
            *[
                MLPMixerLayer(
                    in_dim,
                    num_patches,
                    hidden_units_mlp1,
                    hidden_units_mlp2,
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mlp = nn.Conv2d(in_dim, output_dim, 1)

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / x.norm(dim=2, keepdim=True).mean(1, keepdim=True)
        x = x.unsqueeze(-2) * 0.5
        x = self.conv1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv2(x)
        extra_token_expanded = self.extra_token.expand(
            x.size(0), -1, -1, -1
        )  # Ensure self.extra_token has the same batch size as x

        x = torch.cat([extra_token_expanded, x], dim=3)
        # x = torch.cat([self.extra_token, x], dim=3)
        x = x.permute(0, 3, 2, 1)
        x = self.mlpmixer_blocks(x)
        x = x.permute(0, 3, 2, 1)
        x = x[:, :, :, :1]
        x = self.output_mlp(x)
        x = torch.sigmoid(x)
        return x
