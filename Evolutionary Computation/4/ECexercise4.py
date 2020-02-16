import numpy as np
import random
import copy
import matplotlib.pyplot as plt


def generate_bit_string(l=100):
    bit_string = []
    for i in range(0,l):
        bit_string.append(random.randrange(2))
    return bit_string


def flip_bits(bit_string, p):
    new_string = copy.deepcopy(bit_string)
    for i in range(0, len(bit_string)):
        if np.random.binomial(1,p) == 1:
            new_string[i] = 1- new_string[i]
    return new_string

def compare_strings(bit_string, new_string):
    score_original = sum(bit_string)
    score_new = sum(new_string)
    
    if score_new > score_original:
        return new_string
    else:
        return bit_string

def plot_scores(scores):
    plt.plot(scores)
    plt.ylabel("Amount of 1s")
    plt.xlabel("Nr of iterations")
    plt.show()


def run_algorithm(l, p, nr_iter):
    bit_string = generate_bit_string(l)
    scores = []
    for i in range(nr_iter):
        new_string = flip_bits(bit_string, p)
        bit_string = compare_strings(bit_string, new_string)
        scores.append(sum(bit_string))
    
    plot_scores(scores)
    
    if sum(bit_string) == l:
        print("Maximum achieved")
    else:
        print("Maximum not achieved")


def run_adjusted_algorithm(l, p, nr_iter):
    bit_string = generate_bit_string(l)
    scores = []
    for i in range(nr_iter):
        bit_string = flip_bits(bit_string, p)
        scores.append(sum(bit_string))
    
    plot_scores(scores)
    
    if sum(bit_string) == l:
        print("Maximum achieved")
    else:
        print("Maximum not achieved")

l = 100
p = 1/l
nr_iter = 1500

# Run the standard algorithm ten times 
for i in range(10):
    run_algorithm(l, p, nr_iter)

for i in range(10):
    run_adjusted_algorithm(l, p, nr_iter)
