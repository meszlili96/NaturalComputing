import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt


def plot_results(path):
    kde = cv2.imread(path + "/KDE test.png")
    kde = cv2.cvtColor(kde, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(36, 18))
    plt.imshow(kde)
    plt.show()

    jsd = np.load(path + "/jsd.npy")
    hq_rate = np.load(path + "/hq_rate.npy")
    x_stdev = np.load(path + "/x_stdev.npy")
    ll = np.load(path + "/fs_ll.npy")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(7, 15))

    ax1.plot(jsd)
    ax1.axhline(y=0, color='tab:red')

    ax2.plot(hq_rate)
    ax2.axhline(y=1, color='tab:red')

    ax3.plot(x_stdev)
    ax3.axhline(y=0.02, color='tab:red')

    ax4.plot(ll)

    plt.show()


def compare_kdes(paths, gammas):
    assert len(paths) == len(gammas), "Lengths of paths and gammas should match"
    rows = int(np.ceil(len(paths)/2))

    fig, axis = plt.subplots(rows, 2) #, figsize=(20, 20))
    for k, path in enumerate(paths):
        kde1 = cv2.imread(path + "/KDE test.png")
        j = 0 if (k+1)%2 == 1 else 1
        i = (k+1)//2 + (k+1)%2 - 1
        axis[i, j].imshow(kde1)
        axis[i, j].set_title("Gamma {}".format(gammas[k]))
        axis[i, j].axis(False)

    plt.tight_layout()
    plt.show()


def compare_jsd(paths, gammas):
    assert len(paths) == len(gammas), "Lengths of paths and gammas should match"

    plt.figure()
    for k, path in enumerate(paths):
        jsd = np.load(path + "/jsd.npy")
        plt.plot(jsd, label="{}".format(gammas[k]))

    plt.axhline(y=0, color='k')
    plt.xlabel("Epoch")
    plt.ylabel("JSD(nats)")
    plt.legend()
    plt.show()


def compare_xstdev(paths, gammas):
    assert len(paths) == len(gammas), "Lengths of paths and gammas should match"

    plt.figure()
    for k, path in enumerate(paths):
        x_stdev = np.load(path + "/x_stdev.npy")
        print("Last 20 epochs X stdev range, gamma {} min: {}, max: {}".format(gammas[k],
                                                                               x_stdev[-20:].min(),
                                                                               x_stdev[-20:].max()))
        plt.plot(x_stdev, label="{}".format(gammas[k]))

    plt.axhline(y=0.02, color='k')
    plt.xlabel("Epoch")
    plt.ylabel("Stdev x")
    plt.legend()
    plt.show()


def compare_hq_rate(paths, gammas):
    assert len(paths) == len(gammas), "Lengths of paths and gammas should match"

    plt.figure()
    for k, path in enumerate(paths):
        hq_rate = np.load(path + "/hq_rate.npy")
        print("High quality rate, gamma {}: {}".format(gammas[k], hq_rate[-1]))
        plt.plot(hq_rate, label="{}".format(gammas[k]))

    plt.axhline(y=1, color='k')
    plt.xlabel("Epoch")
    plt.ylabel("High quality rate")
    plt.legend()
    plt.show()


def compare_ll(paths, gammas):
    assert len(paths) == len(gammas), "Lengths of paths and gammas should match"

    plt.figure()
    for k, path in enumerate(paths):
        ll = np.load(path + "/fs_ll.npy")
        plt.plot(ll, label="{}".format(gammas[k]))

    plt.axhline(y=1, color='k')
    plt.xlabel("Epoch")
    plt.ylabel("Real sample log likelihood")
    plt.legend()
    plt.axis([-5, 105, -450, 20])
    plt.show()


def main():
    gammas = [0.0, 0.05, 0.1, 0.4]
    paths = ["../Gamma tuning/8 gauss egan, gamma {}".format(gamma) for gamma in gammas]

    compare_ll(paths, gammas=gammas)

if __name__ == '__main__':
    main()
