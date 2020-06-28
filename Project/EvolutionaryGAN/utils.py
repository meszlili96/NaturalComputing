import numpy as np
import random
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


""" Estimates KDE for a sample
    Parameters:
        sample - sample from a generated distribution
        bandwidth - a user specified bandwidth
    Returns:
        KDE
"""
def sample_kde(sample, bandwidth):
    kernel = KernelDensity(bandwidth=bandwidth)
    kernel.fit(sample)
    return kernel


""" Calculates real data log likelihood under the generator distribution estimated from a sample
    Parameters:
        gen_sample - a sample from a generated distribution
        real_data - a sample of real data 
        bandwidth - a user specified bandwidth
    Returns:
        KDE
"""
def data_log_likelihood(gen_sample, real_data, stdev):
    kernel = sample_kde(gen_sample, stdev/2)
    log_likelihood = kernel.score_samples(real_data)
    return np.average(log_likelihood)


""" Plots a KDE from a sample
    Parameters:
        sample - sample from a generated distribution
        target_distr - a distribution we try to approximate. Needed to get target (x,y) grid and stdev
        results_folder -  a path to save the results
        filename - a specific KDE name
"""
def save_kde(sample, target_distr, results_folder, filename):
    x, y, _ = target_distr.distribution()
    xy = np.vstack([x.ravel(), y.ravel()]).T

    kernel = sample_kde(sample, target_distr.stdev/2)

    z = np.exp(kernel.score_samples(xy)).reshape(x.shape)
    plt.figure()
    plt.imshow(z)
    plt.savefig("{}/KDE {}.png".format(results_folder, filename))

""" Selects the best bandwidth with CV
    Parameters:
        target_distr - a target distribution to sample from 
        bandwidths - an array of bandwidths for CV
"""
def kde_cv(target_distr, bandwidths):
    sample = target_distr.sample(5000)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(sample)
    print("Best bandwidth params: {}".format(grid.best_params_))


def set_seed(seed=99):
    random.seed(30)
    torch.manual_seed(seed)
    np.random.seed(seed)


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

    # test KDE
    output_dir = "utils tests"
    from simdata import EightInCircle, Grid
    eight = EightInCircle(scale=2, stdev=0.05)
    sample = eight.sample(5000)
    save_kde(sample, eight, output_dir, "8 in Circle")
    
    grid = Grid(scale=2, stdev=0.05)
    sample_real = grid.sample(5000)
    save_kde(sample_real, grid, output_dir, "25 in Grid")
    print("Data LL, different distributions: {}".format(data_log_likelihood(sample, sample_real, 0.05)))

    eight2 = EightInCircle(scale=2, stdev=0.02)
    sample_real = eight2.sample(5000)
    print("Data LL, similar distributions: {}".format(data_log_likelihood(sample, sample_real, 0.05)))

    # Eight in circle bandwidth CV
    # 0.05
    #kde_cv(eight, np.linspace(0, 0.5, 41))
    # 0.02
    #kde_cv(eight2, np.linspace(0, 0.4, 41))
    # 0.2
    #kde_cv(EightInCircle(scale=2, stdev=0.2), np.linspace(0, 1, 21))
    # 25 in grid bandwidth CV
    #kde_cv(grid, np.linspace(0, 0.5, 41))


if __name__ == '__main__':
    main()