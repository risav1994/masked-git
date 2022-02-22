from Models.decoder import Decoder
from Models.encoder import Encoder
from Models.codebook import Codebook
from torchsummary import summary
import torch.nn as nn

class VQGan(nn.Module):
    def __init__(self, base_channel=128, levels=[1, 1, 2, 2, 4], num_enc_residual_layers=2, num_dec_residual_layers=3, latent_dims=256, num_vectors=1024):
        super(VQGan, self).__init__()
        self.encoder = Encoder(
            base_channel=base_channel,
            levels=levels,
            num_residual_layers=num_enc_residual_layers,
            latent_dims=latent_dims
        )
        self.decoder = Decoder(
            base_channel=base_channel,
            levels=list(reversed(levels)),
            num_residual_layers=num_dec_residual_layers,
            latent_dims=latent_dims
        )
        self.codebook = Codebook(
            num_vectors=num_vectors,
            embed_size=latent_dims
        )

        self.quant_conv = nn.Conv2d(
            in_channels=latent_dims,
            out_channels=latent_dims,
            kernel_size=1
        )

        self.post_quant_conv = nn.Conv2d(
            in_channels=latent_dims,
            out_channels=latent_dims,
            kernel_size=1
        )


    def forward(self, x):
        inp_encoded = self.encoder(x)
        inp_quant = self.quant_conv(inp_encoded)
        z_q_ma, z_q, z, indices = self.codebook(inp_quant)
        z_q_quant = self.post_quant_conv(z_q_ma)
        inp_decoded = self.decoder(z_q_quant)
        return inp_decoded, z_q_ma, z_q, z, inp_quant

if __name__ == "__main__":
    vqgan = VQGan()
    summary(vqgan, (3, 256, 256))