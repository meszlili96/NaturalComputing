import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import random

class SimulatedDistribution(Enum):
    eight_gaussians = 1
    twenty_five_gaussians = 2


def eight(scale=2.):
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    return centers


def twenty_five(scale=2., size=5):
    x, y = np.meshgrid(np.linspace(-1, 1, size),
                       np.linspace(-1, 1, size))
    centers = [(scale * x, scale * y) for x, y in zip(x.flatten(), y.flatten())]
    return centers


def sample(distribution: SimulatedDistribution, batch_size=1, stdev=0.2):
    if distribution == SimulatedDistribution.eight_gaussians:
        centers = eight()
    elif distribution == SimulatedDistribution.twenty_five_gaussians:
        centers = twenty_five()
    else:
        print("error")

    dataset = []
    for i in range(batch_size):
        point = np.random.randn(2) * stdev
        center = random.choice(centers)
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)

    return dataset


def data_generator(distribution: SimulatedDistribution, batch_size=1, stdev=0.1):
    while True:
        yield sample(distribution, batch_size, stdev)


def gaussians_mixture(centers, grid_size=150, sigma=0.2):
    min_x = min(centers, key=lambda t: t[0])[0] - 3 * sigma
    max_x = max(centers, key=lambda t: t[0])[0] + 3 * sigma
    min_y = min(centers, key=lambda t: t[1])[1] - 3 * sigma
    max_y = max(centers, key=lambda t: t[1])[1] + 3 * sigma
    x, y = np.meshgrid(np.linspace(min_x, max_x, grid_size),
                       np.linspace(min_y, max_y, grid_size))

    g = np.zeros(x.shape)
    for center in centers:
        mu = center
        gc = np.exp(-(((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2.0 * sigma ** 2))) / 2 / np.pi / sigma ** 2
        g += gc

    return g


def test_gereration(distribution: SimulatedDistribution):
    iter = 10000
    x = []
    y = []
    generator = data_generator(distribution)
    for i in range(iter+1):
        sample = next(generator)
        #print(sample)
        x.append(sample[0][0])
        y.append(sample[0][1])

    plt.figure()
    plt.scatter(x, y)
    plt.show()



def plot_sim_distribution():
    g1 = gaussians_mixture(eight())
    g2 = gaussians_mixture(twenty_five())
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(g1)
    plt.subplot(1, 2, 2)
    plt.imshow(g2)
    plt.show()


def main():
    #plot_sim_distribution()
    test_gereration(SimulatedDistribution.twenty_five_gaussians)


if __name__ == '__main__':
    main()
