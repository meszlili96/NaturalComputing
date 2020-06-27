import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

"""Samples noise from unifrom distribution for Generator
"""
def sample_noise(size):
    noise = -1 * torch.rand(size, 2) + 0.5
    return noise


def save_kde(sample, target_distr, results_folder):
    x, y, d = target_distr.distribution()
    xy = np.vstack([x.ravel(), y.ravel()]).T

    bandwidth = 0.2
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(sample)

    z = np.exp(kde.score_samples(xy)).reshape(x.shape)
    plt.figure()
    plt.imshow(z)
    plt.savefig("{}KDE.png".format(results_folder))


def kde_cv(target_distr):
    sample = target_distr.sample(10000)
    bandwidths = np.linspace(0, 1, 20)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(sample)
    print(grid.best_params_)


def set_seed(seed=99):
    torch.manual_seed(99)
    np.random.seed(99)


# KL-divergence for discrete probability distributions
def kl_divergence(p, q):
    # KL-divergence is defined only for when for i: P(i)=0 implies Q(i)=0. If P(i)=0, the corresponding term is interpreted as zero
    diverg = 0
    for i in range(len(p)):
        if p[i] > 0:
            diverg += p[i] * np.log(p[i] / q[i])

    return diverg


# JS-divergence for discrete probability distributions
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def main():
    # JS-divergence test
    # high divergence
    approx_distr = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/128]
    uniform_distr = np.ones(len(approx_distr)) / len(approx_distr)
    js_diver = js_divergence(uniform_distr, approx_distr)
    print("Hight divergence {}".format(js_diver))

    # high divergence
    approx_distr = [1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/16, 3/16]
    uniform_distr = np.ones(len(approx_distr)) / len(approx_distr)
    js_diver = js_divergence(uniform_distr, approx_distr)
    print("Low divergence {}".format(js_diver))

    # same distribution
    js_diver = js_divergence(uniform_distr, uniform_distr)
    print("Same distribution {}".format(js_diver))


if __name__ == '__main__':
    main()