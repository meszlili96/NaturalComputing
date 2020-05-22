import random
from abc import ABCMeta, abstractmethod
from enum import Enum
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, IterableDataset


class SimulatedDistribution(Enum):
    eight_gaussians = 1
    twenty_five_gaussians = 2

class MixtureOfGaussiansDataset(IterableDataset):
    def __init__(self, distribution: SimulatedDistribution):
        super(MixtureOfGaussiansDataset).__init__()
        if distribution == SimulatedDistribution.eight_gaussians:
            self.mixture_of_gaussians = EightInCircle()
        elif distribution == SimulatedDistribution.twenty_five_gaussians:
            self.mixture_of_gaussians = Grid()
        else:
            raise ValueError

    def __iter__(self):
        # TODO: not sure yet if we need to adopt it for workers. All samples are independent
        return self.mixture_of_gaussians.data_generator()


# extracts arrays of x and y from point in sample
# sample - array of 2d points
# x - coordinates on x axis
# x - coordinates on y axis
def extract_xy(sample):
    x, y = [], []

    for point in sample:
        x.append(point[0])
        y.append(point[1])

    return x, y


# abstract class which defines mixture of Gaussians distribution
class MixtureOfGaussians:
    __metaclass__ = ABCMeta

    # stdev - standard deviation of each Gaussian. All Gaussians have the same standard deviation
    # scale - when equals 1, the Gaussians centers are placed inside [-1,1] unit square
    #         other values will scale the square accordingly
    def __init__(self, stdev=0.2, scale=1.):
        self.__stdev = stdev
        self.scale = scale

    # the mixture of Gaussians is defined by number of Gaussians and locations of their centers
    # subclassed have to implement this method and provide the coordinates
    # of centers of all Gaussians included into mixture in scale*[-1,1] square
    @abstractmethod
    def centers(self):
        raise NotImplementedError

    # sample from the distribution
    def sample(self, sample_size=1):
        dataset = []
        for i in range(sample_size):
            mu = random.choice(self.centers())
            cov = [[self.__stdev ** 2, 0], [0, self.__stdev ** 2]]
            point = np.random.multivariate_normal(mu, cov)
            dataset.append(point)

        return dataset

    # generate a stream of data in batches
    def data_generator(self, batch_size=1):
        while True:
            yield self.sample(batch_size)

    # returns (x, y) grid and PDF
    # unit_grid_size - the number of points in a unit (i.e. in [0,1] interval)
    def distribution(self, unit_grid_size=100):
        mus = self.centers()
        # first, determine the grid boundaries
        # by looking for minimum and maximum x and y in Gaussians centers
        # then 3 standard deviatios are added to get correct border
        # because approx 99 % of PDF is inside 3 standard deviatios
        min_x = min(mus, key=lambda t: t[0])[0] - 3 * self.__stdev
        max_x = max(mus, key=lambda t: t[0])[0] + 3 * self.__stdev
        min_y = min(mus, key=lambda t: t[1])[1] - 3 * self.__stdev
        max_y = max(mus, key=lambda t: t[1])[1] + 3 * self.__stdev

        # define number of points for PDF based on borders and unit grid size
        x_grid_size = round(unit_grid_size * (max_x - min_x) / 2)
        y_grid_size = round(unit_grid_size * (max_y - min_y) / 2)

        # make grid
        x, y = np.meshgrid(np.linspace(min_x, max_x, x_grid_size),
                           np.linspace(min_y, max_y, y_grid_size))

        # calculate PDF
        g = np.zeros(x.shape)
        for mu in mus:
            g += np.exp(
                -(((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2.0 * self.__stdev ** 2))) / 2 / np.pi / self.__stdev ** 2

        return x, y, g

    def plot2d(self):
        _, _, g = self.distribution()
        plt.figure()
        plt.imshow(g)
        plt.show()

    def plot3d(self):
        x, y, g = self.distribution()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, g, rstride=1, cstride=1, cmap='jet')
        plt.show()

    # samples the specified number of points and displays scatter plot
    def plot_sample(self, size=10000):
        generator = self.data_generator(batch_size=size)
        sample = next(generator)
        x, y = extract_xy(sample)

        plt.figure()
        plt.scatter(x, y, s=1.5)
        plt.show()


# eight Gaussians arranged in a circle
class EightInCircle(MixtureOfGaussians):
    def centers(self):
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
        # by multiplying on self.scale we adjust the coordinates of the centers
        # to lie inside scale*[-1,1] square
        centers = [(self.scale * x, self.scale * y) for x, y in centers]
        return centers


# evenly spaced squared grid of Gaussians
class Grid(MixtureOfGaussians):
    # size - the number of Gaussians per row and column
    def __init__(self, stdev=0.2, scale=1., size=5):
        super().__init__(stdev, scale)
        self.__size = size

    def centers(self):
        x, y = np.meshgrid(np.linspace(-1, 1, self.__size),
                           np.linspace(-1, 1, self.__size))
        # by multiplying on self.scale we adjust the coordinates of the centers
        # to lie inside scale*[-1,1] square
        centers = [(self.scale * x, self.scale * y) for x, y in zip(x.flatten(), y.flatten())]
        return centers


def main():
    # Demonstration of data generation
    iterable_dataset = MixtureOfGaussiansDataset(SimulatedDistribution.eight_gaussians)
    data_loader = DataLoader(iterable_dataset, batch_size=4)
    for batch in islice(data_loader, 8):
        print(batch)

    # Demonstration of distributions 2d PDFs and 10 000 samples
    fig, axs = plt.subplots(2, 2)
    eight = EightInCircle(scale=2)
    grid = Grid(scale=2)

    _, _, g1 = eight.distribution()
    axs[0, 0].imshow(g1)

    _, _, g2 = grid.distribution()
    axs[0, 1].imshow(g2)

    sample_size = 10000

    generator = eight.data_generator(batch_size=sample_size)
    sample = next(generator)
    x, y = extract_xy(sample)
    axs[1, 0].scatter(x, y, s=1.5)

    generator = grid.data_generator(batch_size=sample_size)
    sample = next(generator)
    x, y = extract_xy(sample)
    axs[1, 1].scatter(x, y, s=1.5)

    plt.show()


if __name__ == '__main__':
    main()
