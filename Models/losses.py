from Models.vgg16 import VGG16
from Models.common_layers import ScalingLayer, NetLinLayer
import torch.nn as nn
import torch

class LPIPS(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512, 512]):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = channels
        self.feature_net = VGG16()
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0], use_dropout=True),
            NetLinLayer(self.channels[1], use_dropout=True),
            NetLinLayer(self.channels[2], use_dropout=True),
            NetLinLayer(self.channels[3], use_dropout=True),
            NetLinLayer(self.channels[4], use_dropout=True)
        ])

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = "Ckpts/vgg.pth"
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)

    def forward(self, real_x, fake_x):
        features_real = self.feature_net(self.scaling_layer(real_x))
        features_fake = self.feature_net(self.scaling_layer(fake_x))
        diffs = {}

        # calc MSE differences between features
        for i in range(len(self.channels)):
            diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2

        return sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.channels))])

def norm_tensor(x):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)

def spatial_average(x):
    return x.mean([2,3], keepdim=True)

def calc_quant_loss(z_q, z, beta=0.25):
    return torch.mean((z_q.detach() - z) ** 2 + beta * (z_q - z.detach()) ** 2)

def adopt_weight(steps, threshold=10000):
    if steps < threshold:
        return 0
    return 1

def calc_lambda(nll_loss, g_loss, vqgan):
    vqgan = vqgan.module if isinstance(vqgan, nn.DataParallel) else vqgan
    last_layer = vqgan.decoder.model[-1].weight
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    λ = torch.clamp(λ, 0, 1e4).detach()
    return 0.2 * λ