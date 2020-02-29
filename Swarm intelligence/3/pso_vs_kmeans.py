import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd


# Returns tuple (data, labels)
def load_iris_dataset():
    iris = datasets.load_iris()
    iris_data = np.array(iris['data'])
    iris_labels = np.array(iris['target'])

    return iris_data, iris_labels


# Returns a class label 0/1 for a data point according to the rule
# 1 if (z1 >= 0.7) or ((z1 <= 0.3) and (z2 >= -0.2 - z1 ))
# 0 otherwise
def artificial_class(data_point):
    if data_point[0] >= 0.7:
        return 1
    else:
        return 1 if data_point[0] <= 0.3 and data_point[1] >= -0.2 - data_point[0] else 0


# Returns tuple (data, labels)
def generate_artificial_data(n_points):
    # draw 400 data point from uniform distribution in interval (-1, 1)
    data_points = np.random.uniform(-1, 1, (n_points, 2))
    class_labels = [artificial_class(x) for x in data_points]
    return data_points, class_labels


def generate_artificial_data2():
    mus = [[-3, 0], [0, 0], [3, 0], [6, 0]]
    sigma = [[0.5, 0.05], [0.05, 0.5]]  # mean and standard deviation
    data = []
    for mu in mus:
        data += np.random.multivariate_normal(mu, sigma, 150).tolist()

    return np.array(data)


def plot_artificial_data(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data0 = np.array([x for ind, x in enumerate(data) if labels[ind] == 0])
    ax.scatter(data0[:, 0], data0[:, 1], alpha=0.8, c='green', edgecolors='none', s=20, label='0')
    data1 = np.array([x for ind, x in enumerate(data) if labels[ind] == 1])
    ax.scatter(data1[:, 0], data1[:, 1], alpha=0.8, c='red', edgecolors='none', s=20, label='1')
    plt.title('Artificial data set')
    plt.legend(loc=2)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()


def plot_artificial_clusters(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cluster in clusters:
        if len(cluster) > 0:
            ax.scatter(cluster[:, 0], cluster[:, 1], alpha=0.8, edgecolors='none', s=20, label='0')

    plt.title('Artificial data clusters')
    plt.legend(loc=2)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


def quantization_error(centroids, clusters):
    n_clusters = len(clusters)
    assert n_clusters == len(centroids), "The number of centroids should match the number of clusters"
    error = 0.0
    for centroid, cluster in zip(centroids, clusters):
        # For some reason PSO tends to create less clusters than requested
        # to prevent this behaviour we penalize empty clusters by returning infinity error for them
        distances_sum = np.sum([euclidean_distance(item, centroid) for item in cluster])
        error += distances_sum/len(cluster) if len(cluster) > 0 else np.inf

    error /= n_clusters
    return error


def cluster_data(data, centroids):
    n_data_points = len(data)
    n_clusters = len(centroids)
    clusters = [[] for _ in range(n_clusters)]

    for j in range(n_data_points):
        min_distance = math.inf
        cluster_ind = None
        data_point = data[j]

        for k in range(n_clusters):
            current_distance = euclidean_distance(data_point, centroids[k])
            if current_distance < min_distance:
                min_distance = current_distance
                cluster_ind = k

        clusters[cluster_ind].append(data_point)

    return np.array([np.array(cluster) for cluster in clusters])


class Particle:

    def __init__(self, position, data, w=0.72, c1=1.49, c2=1.49):
        self.position = position
        self.fitness = quantization_error(position, cluster_data(data, position))
        self.best_position = position.copy()
        self.best_fitness = self.fitness
        # In the paper it is not said how velocities are initialized. We follow lecture slides and initialize them with zeros
        self.velocity = np.zeros(position.shape)
        self.length = self.position.shape[0]
        self.data_dimension = self.position.shape[1]
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update(self, global_best):
        self.update_velocity(global_best)
        self.update_position()

    def update_velocity(self, global_best):
        for i in range(self.length):
            r1, r2 = np.random.rand(self.data_dimension), np.random.rand(self.data_dimension)
            inertia = self.w * self.velocity[i]
            personal_influence = self.c1 * r1 * (self.best_position[i] - self.position[i])
            global_influence = self.c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = inertia + personal_influence + global_influence

    def update_position(self):
        self.position = self.position + self.velocity

    def evaluate(self, data):
        self.fitness = quantization_error(self.position, cluster_data(data, self.position))

        # update local best
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()


class PSO_Clustering:

    def __init__(self, data, n_clusters, n_particles=10, max_iter=1000, verbose=False):
        self.data = data
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.verbose = verbose
        self.global_best = None
        self.global_best_fitness = np.inf
        self.particles = []
        self.init_particles()

    def init_particles(self):
        for i in range(self.n_particles):
            # initialize particles with data randomly sampled data points
            index = np.random.choice(list(range(len(self.data))), self.n_clusters)
            particle = Particle(self.data[index].copy(), self.data)
            self.particles.append(particle)

            # update global best
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best = particle.best_position.copy()

    def fit(self):
        for iteration in range(self.max_iter):
            if self.verbose and iteration % 10 == 0:
                print("PSO iteration {}".format(iteration))

            for particle in self.particles:
                particle.update(self.global_best)
                particle.evaluate(self.data)

            for particle in self.particles:
                # update global best
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best = particle.best_position.copy()

        centroids = self.global_best
        clusters = cluster_data(self.data, centroids)
        return clusters, centroids


def k_means(data, n_clusters, max_iter=1000, verbose=False):
    index = np.random.choice(list(range(len(data))), n_clusters)
    centroids = data[index].copy()

    for iteration in range(max_iter):
        if verbose and iteration % 10 == 0:
            print("K means iteration {}".format(iteration))
        # distribute points between clusters
        clusters = cluster_data(data, centroids)

        # update centroids
        for i in range(n_clusters):
            cluster = clusters[i]
            new_centroid = sum(cluster)/len(cluster)
            centroids[i] = new_centroid

    return clusters, centroids


def main():
    random.seed(99)
    number_of_simulations = 30
    number_of_iterations = 1000

    artificial_data, _ = generate_artificial_data(400)
    k_means_qe_artifical = []
    pso_qe_artifical = []
    for simulation in range(number_of_simulations):
        print('Artificial data simulation {}'.format(simulation+1))
        k_means_clusters, k_means_centroids = k_means(artificial_data, 2, max_iter=number_of_iterations)
        k_means_qe_artifical.append(quantization_error(k_means_centroids, k_means_clusters))

        PSO = PSO_Clustering(artificial_data, 2, max_iter=number_of_iterations)
        pso_clusters, pso_centroids = PSO.fit()
        pso_qe_artifical.append(quantization_error(pso_centroids, pso_clusters))

    data = {'k_means': k_means_qe_artifical, 'pso': pso_qe_artifical}
    df = pd.DataFrame(data=data)
    df.to_csv('artificial_stat.csv')

    k_means_artifical_mean = np.mean(k_means_qe_artifical)
    k_means_artifical_std = np.std(k_means_qe_artifical)
    print('K-means average quantization error {}, std {}, artificial data'.format(k_means_artifical_mean, k_means_artifical_std))

    pso_qe_artifical_mean = np.mean(pso_qe_artifical)
    pso_qe_artifical_std = np.std(pso_qe_artifical)
    print('PSO average quantization error {}, std {}, artificial data'.format(pso_qe_artifical_mean, pso_qe_artifical_std))

    iris_data, _ = load_iris_dataset()
    k_means_qe_iris = []
    pso_qe_iris = []
    for simulation in range(number_of_simulations):
        print('Iris data simulation {}'.format(simulation + 1))
        k_means_clusters, k_means_centroids = k_means(iris_data, 3, max_iter=number_of_iterations)
        k_means_qe_iris.append(quantization_error(k_means_centroids, k_means_clusters))

        PSO = PSO_Clustering(iris_data, 3, max_iter=number_of_iterations)
        pso_clusters, pso_centroids = PSO.fit()
        pso_qe_iris.append(quantization_error(pso_centroids, pso_clusters))

    data = {'k_means': k_means_qe_iris, 'pso': pso_qe_iris}
    df = pd.DataFrame(data=data)
    df.to_csv('iris_stat.csv')

    k_means_qe_iris_mean = np.mean(k_means_qe_iris)
    k_means_qe_iris_std = np.std(k_means_qe_iris)
    print('K-means average quantization error {}, std {}, iris data'.format(k_means_qe_iris_mean, k_means_qe_iris_std))

    pso_qe_iris_mean = np.mean(pso_qe_iris)
    pso_qe_iris_std = np.std(pso_qe_iris)
    print('PSO average quantization error {}, std {}, iris data'.format(pso_qe_iris_mean, pso_qe_iris_std))



if __name__ == "__main__":
    main()