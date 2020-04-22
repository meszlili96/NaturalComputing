import math
import matplotlib.pyplot as plt

def maj_vote_probability(c, p):
    # the first k we consider is c/2 for even number and (c+1)/2 for odd
    k1 = math.ceil(c / 2)
    probability = 0
    # I prefer to work with loops first because it is easier to debug
    for k in range(k1, c+1):
        # we solve a tie in case of even number by tossing a coin, thus for the case when the number of correct
        # decision equals to the number of incorrect decisions we need to multiply the probability to 0.5
        weight = 0.5 if k == k1 and c % 2 == 0 else 1
        probability += weight * math.factorial(c)/math.factorial(k)/math.factorial(c-k)*math.pow(p, k)*math.pow(1-p, c-k)

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
    probability = maj_vote_probability(151, 0.6)
    print("probability {}".format(probability))

    # for students
    graph_p(range(3, 151), 0.6)



if __name__ == "__main__":
    main()