import torch
import torch.nn.functional as F
import torch.nn as nn
import tensorflow as tf

class GroupNorm(nn.Module):
    def __init__(self, in_ch):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=32,
            num_channels=in_ch,
            eps=1e-6,
            affine=True
        )

    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_ch=256, out_ch=256):
        super(ResidualLayer, self).__init__()
        self.layers = []
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.build_model()

    def forward(self, x):
        out = self.model(x)
        if self.in_ch != self.out_ch:
            x = self.up_channel(x)
        return x + out

    def build_model(self):
        in_ch = self.in_ch
        for i in range(2):
            self.layers.append(
                GroupNorm(in_ch)
            )
            self.layers.append(
                Swish()
            )
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=self.out_ch,
                    kernel_size=3,
                    padding=1
                )
            )
            in_ch = self.out_ch
        if self.in_ch != self.out_ch:
            self.up_channel = nn.Conv2d(
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                kernel_size=1
            )
        self.model = nn.Sequential(*self.layers)

class DownSampleLayer(nn.Module):
    def __init__(self, ch=256):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            stride=2,
        )

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)

class UpSampleLayer(nn.Module):
    def __init__(self, ch=256):
        super(UpSampleLayer, self).__init__()
        self.conv =nn.Conv2d(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.conv(x)

class Attention(nn.Module):
    def __init__(self, dims=256):
        super(Attention, self).__init__()
        self.q, self.k, self.v = (
            nn.Conv2d(
                in_channels=dims,
                out_channels=dims,
                kernel_size=1,
            )
            for i in range(3)
        )

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        b, c, h, w, = q.shape
        q_reshaped = q.reshape(b, c, h * w)
        q_reshaped = q_reshaped.permute(0, 2, 1)
        k_reshaped = k.reshape(b, c, h * w)
        v_reshaped = v.reshape(b, c, h * w).permute(0, 2, 1)
        scalar_dot = torch.bmm(q_reshaped, k_reshaped) / (c ** 0.5)
        softmax = F.softmax(scalar_dot, dim=-1)
        attn = torch.bmm(softmax, v_reshaped).permute(0, 2, 1)
        attn_reshaped = attn.reshape(b, c, h, w)
        return attn_reshaped



class NonLocalLayer(nn.Module):
    def __init__(self, dims=256):
        super(NonLocalLayer, self).__init__()
        self.layers = []
        self.dims = dims
        self.build_model()

    def forward(self, x):
        out = self.model(x)
        return x + out

    def build_model(self):
        self.layers.append(
            GroupNorm(self.dims)
        )
        self.layers.append(
            Attention(self.dims)
        )
        self.layers.append(
            nn.Conv2d(
                in_channels=self.dims,
                out_channels=self.dims,
                kernel_size=1
            )
        )
        self.model = nn.Sequential(*self.layers)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale

class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout() if use_dropout else None,
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )
