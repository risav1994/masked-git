from torchsummary import summary
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_layers=3, base_filters=64):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        self.base_filters = base_filters
        self.layers = []
        self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        self.layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.base_filters,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.layers.append(
            nn.LeakyReLU(negative_slope=0.2)
        )
        for i in range(self.num_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels=self.base_filters * (2 ** (min(i, 8))),
                    out_channels=self.base_filters * (2 ** (min(i + 1, 8))),
                    kernel_size=4,
                    stride=2 if i != self.num_layers - 1 else 1,
                    padding=1
                )
            )
            self.layers.append(
                nn.BatchNorm2d(self.base_filters * (2 ** (min(i + 1, 8))))
            )
            self.layers.append(
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.layers.append(
            nn.Conv2d(
                in_channels=self.base_filters * (2 ** (min(self.num_layers, 8))),
                out_channels=1,
                kernel_size=4,
                padding=1
            )
        )
        self.model = nn.Sequential(*self.layers)

if __name__ == "__main__":
    discriminator = Discriminator()
    summary(discriminator, (3, 256, 256))