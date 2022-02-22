from Models.vqgan import VQGan
from Models.discriminator import Discriminator
from Data.generator import DataGen
from Models.losses import calc_quant_loss, adopt_weight, LPIPS, calc_lambda
import torch.nn as nn
import torch.nn.functional as F
import torch

class Train:
    def __init__(self, base_channel=128, levels=[1, 1, 2, 2, 4], num_enc_residual_layers=2, num_dec_residual_layers=3, latent_dims=256, num_vectors=1024, num_layers=3, base_filters=64):
        self.vqgan = VQGan(base_channel, levels, num_enc_residual_layers, num_dec_residual_layers, latent_dims, num_vectors)
        self.discriminator = Discriminator(num_layers, base_filters)
        self.opt_vq = torch.optim.Adam(
            self.vqgan.parameters(),
            lr=1e-4
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4
        )
        self.generator = DataGen()()
        self.perc_loss = LPIPS().eval()

    def __call__(self):
        step = 0
        save_every = 1000
        ckpt_dir = "Ckpts/"
        for img_batch in self.generator:
            imgs = torch.tensor(img_batch).permute(0, 3, 1, 2).contiguous()
            step += 1
            decoded, z_q_ma, z_q, z, inp_quant = self.vqgan(imgs)
            quant_loss = calc_quant_loss(z_q, z)
            disc_real = self.discriminator(imgs)
            disc_fake = self.discriminator(decoded)
            perceptual_loss = self.perc_loss(imgs, decoded)
            rec_loss = torch.abs(imgs - decoded)
            nll_loss = (perceptual_loss + rec_loss).mean()
            g_loss = -torch.mean(disc_fake)

            disc_coeff = adopt_weight(step)
            lmda = 0
            if disc_coeff > 0:
                lmda = calc_lambda(nll_loss, g_loss, self.vqgan)
            vq_loss = nll_loss + quant_loss + disc_coeff * lmda * g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))

            gan_loss = disc_coeff * .5 * (d_loss_real + d_loss_fake)

            self.opt_vq.zero_grad()
            vq_loss.backward(retain_graph=True)

            self.opt_disc.zero_grad()
            gan_loss.backward()

            self.opt_vq.step()
            self.opt_disc.step()

            if step % save_every == 0:
                torch.save(self.vqgan.state_dict(), f"{ckpt_dir}vqgan_{step}.pt")


    def calc_lmda(self, rec_loss, g_loss, tape):
        last_layer = self.vq_model.layers[-1].weights
        rec_grad = tape.gradient(rec_loss, last_layer)[0]
        g_grad = tape.gradient(g_loss, last_layer)[0]
        lmda = tf.norm(rec_grad) / (tf.norm(g_grad) + 1e-6)
        lmda = tf.clip_by_value(lmda, 0, 1e4)
        lmda = tf.stop_gradient(lmda)
        return 0.8 * lmda


if __name__ == "__main__":
    train = Train()
    train()
