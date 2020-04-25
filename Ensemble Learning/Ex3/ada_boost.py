import math
import matplotlib.pyplot as plt
import numpy as np

def ada_boost(err):
    w = 1
    alpha = math.log((1-err)/err)
    w *= math.exp(alpha*err)
    return w

def graph():
    x = [i/1000 for i in range(1,1000)]
    y = [ada_boost(w) for w in x]
    #print(x[np.argmax(y)],max(y))
    #print(x[499],y[499])

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Error of Learner')
    plt.ylabel('Weight')
    plt.axvline(x=0.5, color='r', linestyle='--')
    plt.show()

graph()
