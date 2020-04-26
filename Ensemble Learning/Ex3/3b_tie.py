import math
import matplotlib.pyplot as plt

def weak_prob(c,k,p):
    return math.factorial(c)/math.factorial(k)/math.factorial(c-k)*math.pow(p,k)*math.pow(1-p,c-k)

def maj_vote_probability(w):
    c = 10
    K = math.ceil((w+c)/2)
    s = max(K-w,0)
    probability = 0
    #tie
    if (w+c)%2 == 0:
        #strong classifier correct
        probability += 0.5 * 0.85 * weak_prob(c,s,0.6)
        for k in range(s+1, c+1):
            probability += 0.85 * weak_prob(c,k,0.6)
        #strong classifier incorrect
        if K <= c:
            probability += 0.5 * 0.15 * weak_prob(c,K,0.6)
        for k in range(K+1, c+1):
            probability += 0.15 * weak_prob(c,k,0.6)
    else:
        #strong classifier correct
        for k in range(s, c+1):
            probability += 0.85 * weak_prob(c,k,0.6)
        #strong classifier incorrect
        for k in range(K, c+1):
            probability += 0.15 * weak_prob(c,k,0.6)

    return probability

def graph():
    x = [i for i in range(1, 15)]
    y = [maj_vote_probability(w) for w in x]
    print(max(y))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Weight of Strong Classifier')
    plt.ylabel('Probability')
    plt.show()

graph()
