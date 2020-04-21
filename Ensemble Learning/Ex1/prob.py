import math
import matplotlib.pyplot as plt

def maj_vote_probability(c, p):
    k1 = math.floor(c/2) + 1
    probability = 0
    # I prefer to work with loops first because it is easier to debug
    for k in range(k1, c+1):
        probability += math.factorial(c)/math.factorial(k)/math.factorial(c-k)*math.pow(p,k)*math.pow(1-p,c-k)

    return probability


#def graph_p(c, p_range):


def graph_p(c_range, p):
    probabilities = [maj_vote_probability(c, p) for c in c_range]

    plt.figure()
    plt.plot(c_range, probabilities)
    plt.xlabel('Number of doctors')
    plt.ylabel('Probability')
    plt.title('p = {}'.format(p))
    plt.show()



def main():
    probability = maj_vote_probability(50, 0.6)
    print("probability {}".format(probability))

    # for students
    graph_p(range(1, 101), 0.6)



if __name__ == "__main__":
    main()