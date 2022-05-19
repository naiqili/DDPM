import matplotlib.pyplot as plt
import numpy as np
from numpy import log, exp, pi
from scipy.stats import wishart, gamma
from scipy.stats import multivariate_normal as normal
from numpy.linalg import inv, det
from matplotlib.patches import Ellipse
from scipy.special import loggamma
from numpy import linalg as LA
from utils import mvnrvs


def sample_mu_lam(xs0, ns, ks, k, _mu0, _ka0, _a0, _b0):
    '''
    Sample mu and lambda of the k-th cluster (from normal-gamma distribution)
    xs0: data matrix, mat of N X D
    ns: ns[k] is the size of the k-th cluster, ns[k]==|C_k|, vec of K
    ks: ks[i] is the cluster index of the i-th data, ks[i] == k <-> xs0[i] in C_k, vec of N
    k: sample from the k-th cluster
    _mu0, _ka0, _a0, _b0: hyper priors
    '''
    n = ns[k]
    D = xs0.shape[1]
    xx = xs0[ks == k]
    nn = len(xx)
    mzk = np.mean(xx, axis=0)
    mu_n = (_ka0 * _mu0 + n * mzk) / (_ka0 + n)
    ka_n = _ka0 + n
    a_n = _a0 + n * D / 2
    b_n = _b0
    #     print('b_n1', b_n);
    #     for zi in xs0[ks==k]:
    #         b_n += 0.5*LA.norm(zi-mzk)**2
    b_n += np.sum((xx - np.tile(mzk, (nn, 1))) ** 2)
    #     print('b_n2', b_n);
    b_n += _ka0 * n * LA.norm(mzk - _mu0) ** 2 / (2 * (_ka0 + n))

    lam_k = gamma.rvs(a_n, scale=1 / b_n)
    mu_k = mvnrvs(mu_n, 1 / (ka_n * lam_k))
    #     mu_k = normal.rvs(mu_n, 1/(ka_n*lam_k))
    #     print('a_n', a_n); print('b_n', b_n); print('lam_k', lam_k);
    return lam_k, mu_k


def SSE(xs, K, ks, ns, mu_K, lam_K):
    '''
    Sum of square error, smaller means better
    '''
    sse = 0
    for k in range(K):
        idx = (ks == k)
        for xi in xs[idx]:
            sse += LA.norm(xi - mu_K[k]) ** 2
    return sse


def LLH(xs, K, ks, ns, mu_K, lam_K):
    '''
    Log likelihood for clustering, greater means better
    '''
    llh = 0
    for k in range(K):
        idx = (ks == k)
        for xi in xs[idx]:
            llh += log(normal.pdf(xi, mean=mu_K[k], cov=1 / lam_K[k]))
    return llh