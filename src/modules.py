import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np


class LinearReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.ReLU(True))

    def forward(self, x):
        h = self.f(x)
        return(h)


class SeqNN(nn.Module):
    def __init__(self, num_steps, dim):
        super(SeqNN, self).__init__()
        modules = [
            LinearReLU(dim, dim)
            for _ in range(num_steps)
        ]
        self.f = nn.Sequential(*modules)

    def forward(self, pre_h):
        post_h = self.f(pre_h)
        return(post_h)


class TEncoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(TEncoder, self).__init__()
        self.x2z = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        pre_h = self.x2z(x)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        return(mu, logvar)


class TDecoder(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(TDecoder, self).__init__()
        self.z2x = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        pre_h = self.z2x(z)
        post_h = self.seq_nn(pre_h)
        ld = self.h2ld(post_h)
        correct_ld = self.softplus(ld)
        return(correct_ld)


class tscVAE(nn.Module):
    def __init__(
            self,
            x_dim, xz_dim, #i_dimになってた
            enc_z_h_dim, dec_z_h_dim,
            num_enc_z_layers,
            num_dec_z_layers, **kwargs):
        super(tscVAE, self).__init__()
        self.enc_z = TEncoder(num_enc_z_layers, x_dim, enc_z_h_dim, xz_dim)
        self.dec_z2x = TDecoder(num_enc_z_layers, xz_dim, dec_z_h_dim, x_dim)
        self.softplus = nn.Softplus()
        self.logtheta_x =  Parameter(torch.Tensor(x_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.logtheta_x)

    def forward(self, x):
        # encode z
        qz_mu, qz_logvar = self.enc_z(x)
        qz = dist.Normal(qz_mu, self.softplus(qz_logvar))
        z = qz.rsample()
        # decode z
        xld = self.dec_z2x(z)
        xld = xld + 1e-10
        return(z, qz, xld)


class EEncoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(EEncoder, self).__init__()
        self.x2h = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size,128*128)
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        return(mu, logvar)


class EDecoder(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(EDecoder, self).__init__()
        self.z2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2x = nn.Sequential(
            nn.Linear(h_dim,2048),
            nn.ReLU(),
            nn.Linear(2048,x_dim)
        )
        self.softplus = nn.Softplus()

    def forward(self, z):
        batch_size = z.size(0)
        pre_h = self.z2h(z)
        post_h = self.seq_nn(pre_h)
        post_h = self.h2x(post_h)
        e_feature = self.softplus(post_h)
        return(e_feature)


class escVAE(nn.Module):
    def __init__(
            self,
            i_dim, xz_dim,
            enc_z_h_dim, dec_z_h_dim,
            num_enc_z_layers,
            num_dec_z_layers, **kwargs):
        super(escVAE, self).__init__()
        self.enc_z = EEncoder(num_enc_z_layers, i_dim, enc_z_h_dim, xz_dim)
        self.dec_z2x = EDecoder(num_enc_z_layers, xz_dim, dec_z_h_dim, i_dim)
        self.softplus = nn.Softplus()
        self.logtheta_x =  Parameter(torch.Tensor(i_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.logtheta_x)

    def forward(self, x):
        # encode z
        qz_mu, qz_logvar = self.enc_z(x)
        qz = dist.Normal(qz_mu, self.softplus(qz_logvar))
        z = qz.rsample()
        # decode z
        ld_img = self.dec_z2x(z)
        return(z, qz, ld_img)


class Distributor(nn.Module):
    def __init__(self, latent_dim: int, h_dim: int = 64) -> None:
        super().__init__()
        self.zk2w = nn.Linear(latent_dim, h_dim)
        self.zl2wb = nn.Linear(latent_dim, h_dim+1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, t_z, e_z):
        w_i = self.zk2w(e_z)
        wb_z = self.zl2wb(t_z)
        w_z, b_z = wb_z[..., :-1], wb_z[..., -1]
        lp = -self.logsoftmax(w_i @ w_z.T + b_z)
        return lp

    def inference(self, t_z, val_ez_train=val_ez_train, val_ez = val_ez):
        w_i = self.zk2w(val_ez_train)
        wb_z = self.zl2wb(t_z)
        w_z, b_z = wb_z[..., :-1], wb_z[..., -1]
        calc_z = w_i @ w_z.T + b_z
        calc_z3 = calc_z * 3
        best_z = self.softmax(calc_z3)
        best_ez = val_ez_train.T @ best_z
        return  best_ez
