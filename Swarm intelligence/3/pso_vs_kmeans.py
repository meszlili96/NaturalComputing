import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# PSO constants
w = 0.72
c1 = c2 = 1.49

# Returns tuple (data, labels)
def load_iris_dataset():
    iris = datasets.load_iris()
    iris_data = iris['data'].tolist()
    iris_labels = iris['target']

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
    data_points = np.random.uniform(-1, 1, (n_points, 2)).tolist()
    class_labels = [artificial_class(x) for x in data_points]
    return data_points, class_labels


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


def euclidean_distance(x, y):
    return math.sqrt(sum([(xi - yi)**2 for xi, yi in zip(x, y)]))


def quantization_error(centroids, clusters):
    n_clusters = len(clusters)
    assert n_clusters == len(centroids), "The number of centroids should match the number of clusters"
    error = 0
    for centroid, cluster in zip(centroids, clusters):
        error += sum([euclidean_distance(item, centroid) for item in cluster])/len(cluster) if len(cluster) > 0 else 0

    error /= n_clusters
    return error


class Particle:


    def __init__(self, position):
        self.position = position
        self.best_position = position
        self.best_fitness = math.inf
        self.velocity = np.zeros(position.shape)


    def update_velocity(self, global_best):
        r1 = random.random()
        r2 = random.random()
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best - self.position)


    def update_position(self):
        self.position = self.position + self.velocity


# return tuple of (clusters, centroids)
def pso_clustering(data, n_clusters, n_particles=10, max_iter=1000):
    # randomly initialize particles
    particles = []
    for i in range(n_particles):
        random_sample = random.sample(data, n_clusters)
        particle = Particle(np.array(random_sample))
        particles.append(particle)

    global_best = None
    global_best_fitness = math.inf

    n_data_points = len(data)
    # pso clustering algorithm
    for iter in range(max_iter):
        print("PSO iteration {}".format(iter))
        for particle in particles:
            # initialize clusters
            centroids = particle.position
            clusters = [[] for _ in range(n_clusters)]

            # distribute points between clusters
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

            # calculate fitness
            fitness = quantization_error(centroids, clusters)

            # update local best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.array(centroids)

            # update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best = np.array(centroids)

            # update velocity
            particle.update_velocity(global_best)

            # update position
            particle.update_position()

    centroids = global_best
    clusters = [[] for _ in range(n_clusters)]

    # distribute points between clusters
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

    return clusters, centroids.tolist()


def k_means(data, n_clusters, max_iter=1000):
    centroids = random.sample(data, n_clusters)
    clusters = [[] for _ in range(n_clusters)]

    n_data_points = len(data)
    # pso clustering algorithm
    for iter in range(max_iter):
        print("K means iteration {}".format(iter))
        # distribute points between clusters
        for j in range(n_data_points):
            min_distance = math.inf
            cluster_ind = None
            data_point = data[j]

            for k in range(n_clusters):
                current_distance = euclidean_distance(data_point, centroids[k])
                if current_distance < min_distance:
                    min_distance = current_distance
                    cluster_ind = k

            clusters[cluster_ind].append(np.array(data_point))

        for i in range(n_clusters):
            cluster = clusters[i]
            new_centroid = sum(cluster)/len(cluster)
            centroids[i] = new_centroid

    return clusters, centroids

def main():
    random.seed(9)

    artificial_data, artificial_labels = generate_artificial_data(400)
    # Plot artificial data
    #plot_artificial_data(artificial_data, artificial_labels)

    #clusters1, centroids1 = k_means(artificial_data, 2)
    #fitness1 = quantization_error(centroids1, clusters1)

    #print("K means fitness {}".format(fitness1))

    #clusters2, centroids2 = pso_clustering(artificial_data, 2)
    #fitness2 = quantization_error(clusters2, centroids2)

    iris_data, _ = load_iris_dataset()
    clusters1, centroids1 = k_means(iris_data, 3)
    fitness1 = quantization_error(centroids1, clusters1)
    print("K means fitness {}".format(fitness1))

    clusters2, centroids2 = pso_clustering(iris_data, 3)
    fitness2 = quantization_error(centroids2, clusters2)

    print("PSO fitness {}".format(fitness2))





if __name__ == "__main__":
    main()