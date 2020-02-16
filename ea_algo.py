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
def fitness(l):
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
def selection(l,n):
    m = calc_probabilities(l)
    sorted_m = sorted(m.items(), key=lambda k:k[1])
    return sorted_m[:n]

def crossover_parents(p1,p2):
    crossover_point = random.randint(0, len(p1))
    tmp = p1
    p1 = p2[:crossover_point] + p1[crossover_point:]
    p2 = tmp[:crossover_point] + p2[crossover_point:]
    return p1, p2

def crossover(fittest, samples):
    indices = [k[0] for k in fittest[:2]]
    parents = [samples[i] for i in indices]
    p1, p2 = crossover_parents(parents[0],parents[1])
    indices = [k[0] for k in fittest[2:]]
    parents = [samples[i] for i in indices]
    p3, p4 = crossover_parents(parents[0],parents[1])
    return [p1,p2,p3,p4]

def breed(samples,n):
   return random.sample(samples,len(samples)-n)

def mutate(samples,prob):
    mutated_samples = []
    for s in samples:
        for i in range(len(s)):
            if(random.random() < prob):
                swap_ind = random.randint(0,len(s)-1)

                tmp1 = s[swap_ind]
                tmp2 = s[i]

                s[i] = tmp1
                s[swap_ind] = tmp2
        mutated_samples.append(s)
    return mutated_samples

def genetic_algorithm(towns, generations):
    population=init_population(towns,10)   

    for i in range(generations):
        fit=fitness(population)
        select=selection(fit,4)
        elite=crossover(select,population)
        children=breed(population,4)
        population=mutate(elite+children,0.001)

    best = selection(fitness(population),1)[0]
    best_index = best[0]
    return population[best_index]

#14 cities in Burma (http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/burma14.tsp)
towns=[Coord(16.47,96.10), Coord(16.47,94.44), Coord(20.09,92.54), Coord(22.39,93.37), Coord(25.23,97.24), Coord(22.00,96.05), Coord(20.47,97.02), Coord(17.20,96.29), Coord(16.30,97.38), Coord(14.05,98.12), Coord(16.53,97.38), Coord(21.52,95.59),Coord(19.41,97.13),Coord(20.09,94.55)]

best_route_coords=genetic_algorithm(towns,10)
best_route=[]
for i in range(len(best_route_coords)):
    best_route.append(towns.index(best_route_coords[i]))
print(best_route)
