import random
from abc import ABCMeta, abstractmethod
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, IterableDataset
import torch.nn as nn
from utils import js_divergence

# To add a new distribution subclass MixtureOfGaussians and specify Gaussians centers in a unit square
# Then add a new case to SimulatedDistribution enum and expand MixtureOfGaussiansDataset with it
class SimulatedDistribution(Enum):
    eight_gaussians = 1
    twenty_five_gaussians = 2
    standard_gaussian = 3

class MixtureOfGaussiansDataset(IterableDataset):
    def __init__(self, distribution: SimulatedDistribution, stdev=0.2, scale=1., length=None):
        super(MixtureOfGaussiansDataset).__init__()
        if distribution == SimulatedDistribution.eight_gaussians:
            self.distribution = EightInCircle(stdev=stdev, scale=scale, length=length)
        elif distribution == SimulatedDistribution.twenty_five_gaussians:
            self.distribution = Grid(stdev=stdev, scale=scale,length=length)
        elif distribution == SimulatedDistribution.standard_gaussian:
            self.distribution = StandardGaussian(stdev=stdev, scale=scale, length=length)
        else:
            raise ValueError

        self.data_generator = self.distribution.data_generator()
        self.length = length

    def __iter__(self):
        # TODO: not sure yet if we need to adopt it for workers. All samples are independent
        return self.data_generator

    def __len__(self):
        return self.length


# extracts arrays of x and y from point in sample
# sample - array of 2d points
# x - coordinates on x axis
# y - coordinates on y axis
def extract_xy(sample):
    x, y = [], []

    for point in sample:
        x.append(point[0])
        y.append(point[1])

    return x, y

def save_sample(sample, img_name):
    x, y = extract_xy(sample)

    plt.figure()
    plt.scatter(x, y, s=1.5)
    plt.savefig(img_name)
    plt.close()

# abstract class which defines mixture of Gaussians distribution
class MixtureOfGaussians:
    __metaclass__ = ABCMeta

    # stdev - standard deviation of each Gaussian. All Gaussians have the same standard deviation
    # scale - when equals 1, the Gaussians centers are placed inside [-1,1] unit square
    #         other values will scale the square accordingly
    def __init__(self, stdev=0.2, scale=1., length=None):
        self.stdev = stdev
        self.scale = scale
        self.length = length

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
            cov = [[self.stdev ** 2, 0], [0, self.stdev ** 2]]
            point = np.random.multivariate_normal(mu, cov)
            dataset.append(point)

        return np.array(dataset, np.float32)

    # generate a stream of data in batches
    def data_generator(self, batch_size=1):
        if self.length is not None:
            yielded = 0
            while yielded < self.length:
                yielded += 1
                yield self.sample(batch_size)
        else:
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
        min_x = min(mus, key=lambda t: t[0])[0] - 3 * self.stdev
        max_x = max(mus, key=lambda t: t[0])[0] + 3 * self.stdev
        min_y = min(mus, key=lambda t: t[1])[1] - 3 * self.stdev
        max_y = max(mus, key=lambda t: t[1])[1] + 3 * self.stdev

        # define number of points for PDF based on borders and unit grid size
        x_grid_size = int(round(unit_grid_size * (max_x - min_x) / 2))
        y_grid_size = int(round(unit_grid_size * (max_y - min_y) / 2))

        # make grid
        x, y = np.meshgrid(np.linspace(min_x, max_x, x_grid_size),
                           np.linspace(min_y, max_y, y_grid_size))

        # calculate PDF
        g = np.zeros(x.shape)
        for mu in mus:
            g += np.exp(
                -(((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2.0 * self.stdev ** 2))) / 2 / np.pi / self.stdev ** 2

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

    def likelihood_of(self, sample):
        mus = self.centers()
        log_likelihood = 1
        for item in sample:
            item_likelihood = 0
            for mu in mus:
                item_likelihood += np.exp(-(((item[0] - mu[0]) ** 2 + (item[1] - mu[1]) ** 2) / (2.0 * self.stdev ** 2))) / 2 / np.pi / self.stdev ** 2
            log_likelihood += np.log(item_likelihood)
        return log_likelihood

    """Calculates metrics used in the paper https://arxiv.org/pdf/1811.11357.pdf to evaluate Generator performance:
            Parameters:
                sample - a sample from Generator
            Returns:
                hq_samples_percentage - a percentage of high quality samples (assigned to a mode)
                stdev - a tuple (stdev_x, stdev_y), standard deviations of each component of 2D samples
                js_diver - Jensen-Shannon divergence between the sample mode distribution and a uniform distribution
         
    """
    def measure_sample_quality(self, sample):
        modes = self.centers()

        # none key is for low quality samples which are not assigned to any mode
        unassigned_key = "none"
        sample_distr = {unassigned_key: []}
        for idx in range(len(modes)):
            sample_distr[idx] = []

        for point in sample:
            assigned = False
            for idx, mode in enumerate(modes):
                dist = np.linalg.norm(point - mode)
                if dist < 4 * self.stdev:
                    assigned = True
                    mode_samples = sample_distr[idx]
                    mode_samples.append(point)
                    sample_distr[idx] = mode_samples
                    break

            if not assigned:
                unassigned_samples = sample_distr[unassigned_key]
                unassigned_samples.append(point)
                sample_distr[unassigned_key] = unassigned_samples

        assigned_samples_num = len(sample) - len(sample_distr[unassigned_key])
        hq_samples_percentage = assigned_samples_num / len(sample)

        # compute stdev for each mode
        stdevs = []
        # the distribution of assigned generated samples over modes
        mode_distr = np.zeros(len(modes))
        for mode_id, samples in sample_distr.items():
            if mode_id is not unassigned_key and len(samples) > 0:
                mode = modes[mode_id]
                # since we want to measure a spread around the real mode, we use real mode as sample mean
                mncn_samples = samples - np.asarray(mode)
                # we are not interested in covariance, so we calculate variance by component
                # In denominator N-1 is used since we work with a sample
                stdev = np.sqrt(np.sum((mncn_samples**2), axis=0)/len(samples))
                stdevs.append(stdev)

                mode_distr[mode_id] = len(samples)/assigned_samples_num

        #average stdev by component
        stdev = np.mean(stdevs, axis=0) if len(stdevs) > 0 else [1, 1]

        # for mixture of Gaussians all modes are equally likely
        uniform_distr = np.ones(len(modes))/len(modes)
        # calculate Jensen-Shannon divergence
        js_diver = js_divergence(mode_distr, uniform_distr)

        return hq_samples_percentage, stdev, js_diver


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
    def __init__(self, stdev=0.2, scale=1., size=5, length=None):
        super().__init__(stdev, scale, length)
        self.__size = size

    def centers(self):
        x, y = np.meshgrid(np.linspace(-1, 1, self.__size),
                           np.linspace(-1, 1, self.__size))
        # by multiplying on self.scale we adjust the coordinates of the centers
        # to lie inside scale*[-1,1] square
        centers = [(self.scale * x, self.scale * y) for x, y in zip(x.flatten(), y.flatten())]
        return centers


# evenly spaced squared grid of Gaussians
class StandardGaussian(MixtureOfGaussians):
    def __init__(self, stdev=0.2, scale=1., center=(0, 0), length=None):
        super().__init__(stdev, scale, length)
        self.__center = center

    def centers(self):
        return [self.__center]

def main():
    # Demonstration of data generation
    batch_size = 4
    iterable_dataset = MixtureOfGaussiansDataset(SimulatedDistribution.standard_gaussian, length=8 * batch_size)
    data_loader = DataLoader(iterable_dataset, batch_size=4)
    for n_batch, batch in enumerate(data_loader):
        print(batch)

    # Test sample quality calculation
    eight = EightInCircle(scale=2, stdev=0.05)
    # Generate 80% high quality samples uniform distr, low std
    sample = []
    # for each mode 400 HG samples and 100 bad samples
    for mode in eight.centers():
        stdev = 0.03
        hq_points = np.random.multivariate_normal(mode, [[stdev**2, 0], [0, stdev**2]], 400)
        sample.extend(hq_points)

        lq_points = np.random.multivariate_normal(mode, [[stdev ** 2, 0], [0, stdev ** 2]], 100) + [0.5, 0.5]
        sample.extend(lq_points)
    hq_percenage, stdev, js_diver = eight.measure_sample_quality(np.array(sample))
    print("For a good balanced sample {} HG samples rate, stdev {}, JS divergence {}".format(hq_percenage, stdev, js_diver))

    # Generate 80% high quality samples non uniform distr, low std
    sample = []
    # for each mode 400 HG samples and 100 bad samples
    for mode in eight.centers()[:2]:
        stdev = 0.03
        hq_points = np.random.multivariate_normal(mode, [[stdev ** 2, 0], [0, stdev ** 2]], 1600)
        sample.extend(hq_points)

        lq_points = np.random.multivariate_normal(mode, [[stdev ** 2, 0], [0, stdev ** 2]], 400) + [0.5, 0.5]
        sample.extend(lq_points)
    hq_percenage, stdev, js_diver = eight.measure_sample_quality(np.array(sample))
    print("For a good unbalanced sample {} HG samples rate, stdev {}, JS divergence {}".format(hq_percenage, stdev, js_diver))

    # Generate 20% high quality samples non uniform distr, low std
    sample = []
    # for each mode 400 HG samples and 100 bad samples
    for idx, mode in enumerate(eight.centers()):
        stdev = 0.03
        hq_points = np.random.multivariate_normal(mode, [[stdev ** 2, 0], [0, stdev ** 2]], 5 if idx < 6 else idx*50)
        sample.extend(hq_points)

        lq_points = np.random.multivariate_normal(mode, [[stdev ** 2, 0], [0, stdev ** 2]], 400) + [0.5, 0.5]
        sample.extend(lq_points)
    hq_percenage, stdev, js_diver = eight.measure_sample_quality(np.array(sample))
    print(
        "For a bad unbalanced sample {} HG samples rate, stdev {}, JS divergence {}".format(hq_percenage, stdev, js_diver))

    # Demonstration of distributions 2d PDFs and 10 000 samples
    fig, axs = plt.subplots(2, 2)
    eight = EightInCircle(scale=2, stdev=0.02)
    grid = Grid(scale=2, stdev=0.05)

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
