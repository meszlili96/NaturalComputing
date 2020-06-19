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


