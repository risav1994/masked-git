from Models.common_layers import ResidualLayer, UpSampleLayer, NonLocalLayer, GroupNorm
from torchsummary import summary
import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, base_channel=128, levels=[4, 2, 2, 1, 1], num_residual_layers=3, latent_dims=256):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.base_channel = base_channel
        self.num_residual_layers = num_residual_layers
        self.levels = levels
        self.layers = []
        self.build_model()

    def __call__(self, x):
        return self.model(x)

    def build_model(self):
        residual_channel = self.base_channel * self.levels[0]
        self.layers.append(
            nn.Conv2d(
                in_channels=self.latent_dims,
                out_channels=residual_channel,
                kernel_size=3,
                padding=1
            )
        )
        self.layers.append(
            ResidualLayer(
                in_ch=residual_channel,
                out_ch=residual_channel
            )
        )
        self.layers.append(
            NonLocalLayer(
                dims=residual_channel
            )
        )
        self.layers.append(
            ResidualLayer(
                in_ch=residual_channel,
                out_ch=residual_channel
            )
        )
        residual_levels = self.levels
        in_channels = residual_channel
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
                if idx == 0:
                    self.layers.append(
                        NonLocalLayer(
                            dims=residual_channel
                        )
                    )
            if idx != len(residual_levels) - 1:
                self.layers.append(
                    UpSampleLayer(
                        ch=residual_channel
                    )
                )
        self.layers.append(
            GroupNorm(residual_channel)
        )

        self.layers.append(
            nn.Conv2d(
                in_channels=residual_channel,
                out_channels=3,
                kernel_size=3,
                padding=1
            )
        )
        self.model = nn.Sequential(*self.layers)


if __name__ == "__main__":
    decoder = Decoder()
    summary(decoder, (256, 16, 16))