import torch
import torch.nn as nn
from encoder import Encoder

class VectorPosNet(nn.Module):
    """Neural network for predicting vector positions"""
    def __init__(self, cond_dim, hidden_dim, pos_dim):
        super(VectorPosNet, self).__init__()

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(pos_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pos_dim),
        )

    def forward(self, pos, time, cond):
        time_emb = self.time_embedding(time)
        cond_emb = self.cond_embedding(cond)
        x = torch.cat([pos, time_emb, cond_emb], dim=1)
        x = self.net(x)
        return x
    

class AtomDiffusion(nn.Module):
    """Diffusion model for atoms"""
    def __init__(self, encoder: Encoder, denoiser: VectorPosNet, T=100, beta_1=1e-4, beta_T=2e-2):
        super(AtomDiffusion, self).__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T

        self.betas = torch.linspace(beta_1, beta_T, T+1)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self,x0,t,epsilon):
        mean = torch.sqrt(self.alpha_bars[t]) * x0
        std = torch.sqrt(1.0 - self.alpha_bars[t])
        return mean + std * epsilon
    
    def reverse_diffusion(self,x_t,t,epsilon, y):
        mean =  1./torch.sqrt(self.alpha[t]) * (x_t - (self.beta[t])/torch.sqrt(1-self.alpha_bar[t])*self.network(x_t, t, y)) 
        std = torch.where(t>0, torch.sqrt(((1-self.alpha_bar[t-1]) / (1-self.alpha_bar[t]))*self.beta[t]), 0)
        
        return mean + std*epsilon

    #TODO sampling loss


