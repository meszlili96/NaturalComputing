import math
import matplotlib.pyplot as plt

def maj_vote_probability(w):
    c = 10
    K = math.floor((w+c)/2 + 1)
    s = max(K - w, 0)
    probability = 0
    #strong classifier correct
    for k in range(s, c+1):
        probability += 0.85 * math.factorial(c)/math.factorial(k)/math.factorial(c-k)*math.pow(0.6,k)*math.pow(0.4,c-k)
    #strong classifier incorrect    
    for k in range(K, c+1):
        probability += 0.15 *math.factorial(c)/math.factorial(k)/math.factorial(c-k)*math.pow(0.6,k)*math.pow(0.4,c-k)

    return probability

def graph():
    x = [i for i in range(1, 15)]
    y = [maj_vote_probability(w) for w in x]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Weight of Strong Classifier')
    plt.ylabel('Probability')
    plt.show()

graph()
