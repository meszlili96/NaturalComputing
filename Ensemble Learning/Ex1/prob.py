import math
import matplotlib.pyplot as plt
import operator as op
from functools import reduce


# I don't understand this code completely, took it from
# https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
# This if the faster version to compute a combination and more stable to overflow
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def maj_vote_probability(c, p):
    # the first k we consider is c/2 for even number and (c+1)/2 for odd
    k1 = math.ceil(c / 2)
    probability = 0
    # I prefer to work with loops first because it is easier to debug
    for k in range(k1, c+1):
        # we solve a tie in case of even number by tossing a coin, thus for the case when the number of correct
        # decision equals to the number of incorrect decisions we need to multiply the probability to 0.5
        weight = 0.5 if k == k1 and c % 2 == 0 else 1
        probability += weight * ncr(c, k) * math.pow(p, k)*math.pow(1-p, c-k)

    return probability


def plot_p_and_c(p_values=[0.4, 0.49, 0.5, 0.51, 0.6, 0.75, 0.9], c_range=range(1, 101)):
    plt.figure()

    for p in p_values:
        probabilities = [maj_vote_probability(c, p) for c in c_range]
        linestyle = '--' if p==0.5 else '-'
        plt.plot(c_range, probabilities, linestyle=linestyle, label='p = {}'.format(p))

    plt.xlabel('Size of ensemble')
    plt.ylabel('Probability of correct majority vote')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.subplots_adjust(right=0.75)
    plt.show()


def plot_students(c_range=range(1, 101)):
    p = 0.6
    probabilities = [maj_vote_probability(c, p) for c in c_range]
    plt.figure()
    plt.plot(c_range, probabilities, label='p = {}'.format(p))
    plt.xlabel('Number of students')
    plt.ylabel('Probability of correct majority vote')
    plt.show()



def main():
    probability = maj_vote_probability(39, 0.6)
    print("probability {}".format(probability))

    plot_p_and_c()



if __name__ == "__main__":
    main()