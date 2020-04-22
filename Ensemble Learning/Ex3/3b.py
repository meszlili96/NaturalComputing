import math
import matplotlib.pyplot as plt

def maj_vote_probability(w):
    probability = w * 0.85 * (11-w)/10 * math.factorial(10)/math.factorial(5)/math.factorial(5)*math.pow(0.6, 5)*math.pow(0.4, 5)
    for k in range(6, 11):
        probability += (11-w)/10 * math.factorial(10)/math.factorial(k)/math.factorial(10-k)*math.pow(0.6, k)*math.pow(0.4, 10-k)

    return probability

def graph():
    x = [i for i in range(12)]
    y = [maj_vote_probability(w) for w in x]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Weight of Strong Classifier')
    plt.ylabel('Probability')
    plt.show()

graph()
