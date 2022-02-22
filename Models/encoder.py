from torchsummary import summary
from Models.common_layers import ResidualLayer, DownSampleLayer, NonLocalLayer, GroupNorm, Swish
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, base_channel=128, levels=[1, 1, 2, 2, 4], num_residual_layers=2, latent_dims=256):
        super(Encoder, self).__init__()
        self.base_channel = base_channel
        self.num_residual_layers = num_residual_layers
        self.latent_dims = latent_dims
        self.levels = levels
        self.layers = []
        self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        self.layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.base_channel,
                kernel_size=3,
                padding=1
            )
        )
        residual_levels = self.levels
        in_channels = self.base_channel
        for idx, level in enumerate(residual_levels):
            residual_channel = self.base_channel * level
            for _ in range(self.num_residual_layers):
                self.layers.append(
                    ResidualLayer(
                        in_ch=in_channels,
                        out_ch=residual_channel
                    )
                )
                in_channels = residual_channel
                if idx == len(residual_levels) - 1:
                    self.layers.append(
                        NonLocalLayer(
                            dims=residual_channel
                        )
                    )
            if idx != len(residual_levels) - 1:
                self.layers.append(
                    DownSampleLayer(
                        ch=residual_channel
                    )
                )
        residual_channel = self.base_channel * self.levels[-1]
        self.layers.append(
            ResidualLayer(
                in_ch=in_channels,
                out_ch=residual_channel
            )
        )
        in_channels = residual_channel
        self.layers.append(
            NonLocalLayer(
                dims=residual_channel
            )
        )
        self.layers.append(
            ResidualLayer(
                in_ch=in_channels,
                out_ch=residual_channel
            )
        )
        self.layers.append(
            GroupNorm(residual_channel)
        )
        self.layers.append(
            Swish()
        )
        self.layers.append(
            nn.Conv2d(
                in_channels=residual_channel,
                out_channels=self.latent_dims,
                kernel_size=3,
                padding=1
            )
        )
        self.model = nn.Sequential(*self.layers)


if __name__ == "__main__":
    device = torch.device("cpu")
    encoder = Encoder().to(device)
    summary(encoder, (3, 256, 256))
