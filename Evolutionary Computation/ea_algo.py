import math
import random

class Coord:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    #distance between two coordinates
    def dist(self, coord):
        return math.sqrt(pow((self.x-coord.x),2)+pow((self.y-coord.y),2))


#initialise n number of candidates from list l
def init_population(l,n):
    samples_list = []
    for i in range(n):
        sample = random.sample(l, len(l))
        samples_list.append(sample)
    return samples_list

#calculate the fitness of candidate c
def calc_fitness(c):
    s = 0
    for i in range(1,len(c)):
        s += c[i-1].dist(c[i])
    return s

#calculate the fitness of each candidate from list l
def calc_population_fitness(l):
    fitness_list = []    
    for i in l:
        fitness_list.append(calc_fitness(i))
    return fitness_list

#calculate the probability of each candidate from list l
def calc_probabilities(l):
    prob_map = {}
    sum_l = sum(l)
    for i in range(len(l)):
        prob_map[i]=l[i]/sum_l
    return prob_map

#select n number of fittest candidates from list l
def select_fittest(l,n):
    m = calc_probabilities(l)
    sorted_m = sorted(m.items(), key=lambda k:k[1])
    return sorted_m[len(sorted_m)-n:]

#def crossover():


def genetic_algorithm():
    sampling=init_population(towns,4)
    s=calc_population_fitness(sampling)
    print(select_fittest(s,2))
    print(select_fittest(s,3))
#    while
#        select_fittest(l)
#        crossover
#        s=calc_population_fitness(sampling)


#14 cities in Burma (http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/burma14.tsp)
towns=[Coord(16.47,96.10), Coord(16.47,94.44), Coord(20.09,92.54), Coord(22.39,93.37), Coord(25.23,97.24), Coord(22.00,96.05), Coord(20.47,97.02), Coord(17.20,96.29), Coord(16.30,97.38), Coord(14.05,98.12), Coord(16.53,97.38), Coord(21.52,95.59),Coord(19.41,97.13),Coord(20.09,94.55)]

genetic_algorithm()
