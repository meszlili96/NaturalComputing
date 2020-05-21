import numpy as np
import matplotlib.pyplot as plt

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


def gaussians(centers, grid_size=150, sigma=0.2):
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

def main():
    g1 = gaussians(eight())
    g2 = gaussians(twenty_five())
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(g1)
    plt.subplot(1, 2, 2)
    plt.imshow(g2)
    plt.show()

if __name__ == '__main__':
    main()
