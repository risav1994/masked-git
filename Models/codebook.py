from torchsummary import summary
import torch.nn as nn
import torch

class Codebook(nn.Module):
    def __init__(self, num_vectors=1024, embed_size=256):
        super(Codebook, self).__init__()
        self.embedding = nn.Embedding(num_vectors, embed_size)
        self.embedding.weight.data.uniform_(-1.0 / num_vectors, 1.0 / num_vectors)
        self.embed_size = embed_size
        self.num_vectors = num_vectors

    def forward(self, x):
        z = x.permute(0, 2, 3, 1).contiguous()
        z_reshaped = z.view(-1, 1, self.embed_size).tile((1, self.num_vectors, 1))
        weight = torch.unsqueeze(self.embedding.weight, 0)
        d = (z_reshaped - weight) ** 2
        d = torch.sum(d, dim=-1)
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices).view(z.shape)
        z_q_ma = z + (z_q - z).detach()
        z_q_ma = z_q_ma.permute(0, 3, 1, 2)
        return z_q_ma, z_q, z, indices

if __name__ == "__main__":
    codebook = Codebook()
    summary(codebook, (256, 16, 16))
