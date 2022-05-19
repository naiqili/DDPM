import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Distribution, Uniform
from torch.distributions.multivariate_normal import MultivariateNormal


class CouplingLayer(nn.Module):
    """
  Implementation of the additive coupling layer from section 3.2 of the NICE
  paper.
  """

    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2, logdet

        # Inverse additive coupling layer
        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    """
  Implementation of the scaling layer from section 3.3 of the NICE paper.
  """

    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)

        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian


class LogisticDistribution(Distribution):
    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size).to(self.device_name)

        return torch.log(z) - torch.log(1. - z)


class GaussianDistribution(Distribution):
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -0.5 * (x ** 2)  # + constant


# return -0.5*torch.sum(x**2) # + constant

#   def sample(self, size):
#     if USE_CUDA:
#       z = Uniform(torch.cuda.FloatTensor([0.]), torch.cuda.FloatTensor([1.])).sample(size)
#     else:
#       z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

#     return torch.log(z) - torch.log(1. - z)

class ClusterDistribution(Distribution):
    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name

    def log_prob(self, x, K, ks, ns, mu_K, lam_K, oodm=20, oodthresh_perc=0.01):
        x = x.to(self.device_name)

        D = x.shape[1]
        N = x.shape[0]
        oodthresh = N*oodthresh_perc
        llh = 0
#         llh_g = torch.sum(-(F.softplus(x) + F.softplus(-x)))
#         llh += reglam * llh_g
        Klst = list(range(K))
        #     random.shuffle(Klst)
        #
        oodthresh = min(oodthresh, max([np.sum(ks == k) for k in Klst]))
#         print('ood',oodthresh)
        for k in Klst:
            cs_index = torch.from_numpy(ks == k).to(self.device_name)
            _n = torch.sum(cs_index).to(self.device_name)
            if _n < oodthresh:
                continue
            # if _n == 0: continue
            xk = x[cs_index]
#             print(xk.requires_grad)
            perm = [i for i in torch.randperm(x.shape[0]) if (ns[ks[i]] >= oodthresh and ks[i] != k)]
            idx = np.asarray(perm[:oodm])
            negxk = x[idx]
            cov = torch.diag(torch.tensor([1 / np.sqrt(lam_K[k])] * D, requires_grad=False)).to(self.device_name).double()
            m = MultivariateNormal(torch.tensor(mu_K[k], requires_grad=False).to(self.device_name), scale_tril=cov)
            _llh_pos = m.log_prob(xk) / (_n * D)
            _llh_neg = m.log_prob(negxk) / (len(idx) * D)
            llh += torch.sum(_llh_pos) + math.log(ns[k])
            
            # 拉近簇内，拉远负样本
#             llh += torch.clamp(- contra_lam * torch.sum(_llh_neg), max=hinge) 
#             llh += torch.sum(_llh_pos) - torch.clamp(contra_lam * torch.sum(_llh_neg),
#                                                      min=hinge) + -0.5 * reglam * torch.sum(xk ** 2) / (_n * D)
        return llh


class NICE(nn.Module):
    def __init__(self, data_dim, num_coupling_layers=6, num_hidden_units=512, num_net_layers=2, device_name='cpu'):
        super().__init__()

        self.data_dim = data_dim
        self.device_name = device_name

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                 for i in range(num_coupling_layers)]

        #     masks = self._get_random_mask(data_dim, num_coupling_layers)

        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                                            hidden_dim=num_hidden_units,
                                                            mask=masks[i], num_layers=num_net_layers)
                                              for i in range(num_coupling_layers)])
        self.scaling_layer = ScalingLayer(data_dim=data_dim)

        self.prior = ClusterDistribution(self.device_name)
        #     self.pre_prior = LogisticDistribution()
        self.pre_prior = GaussianDistribution()

    def forward(self, x, K, ks, ns, mu_K, lam_K, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x)
            log_likelihood = self.prior.log_prob(z, K, ks, ns, mu_K, lam_K) + log_det_jacobian
            return z, log_likelihood

        return self.f_inverse(x)

    def pre_forward(self, x, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x)
            log_likelihood = torch.sum(self.pre_prior.log_prob(z), dim=1) + log_det_jacobian
            return z, log_likelihood

        return self.f_inverse(x)

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        # z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z):
        x = z
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, log_det_jacobian = coupling_layer(x, 0, invert=True)
        return x

    #   def sample(self, num_samples):
    #     z = self.prior.sample([num_samples, self.data_dim]).view(self.samples, self.data_dim)
    #     return self.f_inverse(z)

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask  # flip mask orientation
        mask = torch.tensor(mask).requires_grad_(False)
        mask = mask.to(self.device_name)
        return mask.float()

    def _get_random_mask(self, data_dim, num_coupling_layers):
        res = []
        for i in range(num_coupling_layers):
            if i % 2 == 0:
                mask = np.zeros(data_dim)
                idx = np.arange(data_dim)
                np.random.shuffle(idx)
                idx = idx[:data_dim // 2]
                mask[idx] = 1.
                mask = torch.tensor(mask).requires_grad_(False)
                mask = mask.to(self.device_name)
            else:
                mask = res[-1].clone()
                mask = 1.0 - mask
            res.append(mask.float())
        return res