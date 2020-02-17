import math
import random
import copy
import sys
import matplotlib.pyplot as plt

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
    crossover_point1 = random.randint(0, len(p1)-1)
    crossover_point2 = random.randint(0, len(p1)-1)
    start = min(crossover_point1,crossover_point2)
    end = max(crossover_point1,crossover_point2)
    child1 = p2[start:end]
    child2 = [i for i in p2 if i not in child1]
    return child1+child2

def crossover(fittest, samples):
    indices = [k[0] for k in fittest[:2]]
    parents = [samples[i] for i in indices]
    fit1 = crossover_parents(parents[0],parents[1])
    indices = [k[0] for k in fittest[2:]]
    parents = [samples[i] for i in indices]
    fit2 = crossover_parents(parents[0],parents[1])
    return [fit1,fit2]

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

def opt_2(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)):
                if (j-i == 1):
                    continue
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1]
                if (calc_fitness(new_route) < calc_fitness(best)):
                    best = new_route
                    improved = True
            route = best
    return best

def opt_2_all(samples):
    improved_samples = []
    for s in samples:
        improved_samples.append(opt_2(s))
    return improved_samples

def genetic_algorithm(orig_coords, generations, memetic=False):
    population=init_population(orig_coords,10)
    best = []

    for i in range(generations):
        fit=fitness(population)
        select=selection(fit,4)
        crossover_children=crossover(select,population)
        children=breed(population,2)
        population=mutate(crossover_children+children,0.001)
        if memetic:
            population=opt_2_all(population)
        best_tuple = selection(fitness(population),1)[0]
        best_index = best_tuple[0]
        #print_results(population[best_index],orig_coords)
        best.append(calc_fitness(population[best_index]))

    return best

def print_results(best_route_coords,orig_coords):
    best_route=[]
    for i in range(len(best_route_coords)):
        best_route.append(orig_coords.index(best_route_coords[i]))
    print("Order of locations: " + str(best_route) + ", value: " + str(calc_fitness(best_route_coords)))

def plot_statistics(best, best_mem):
    x = [i for i in range(len(best))]
    y = best
    plt.plot(x, y, label = "Simple")

    y_mem = best_mem
    plt.plot(x, y_mem, label = "Mematic")

    plt.xlabel('generations') 
    plt.ylabel('distance') 
    plt.title('Comparison of simple EA and Memetic Algorithm')
    plt.legend()

    plt.show()

def read_file(txt):
    coords = []
    f = open(txt, "r")
    read = False
    for line in f:
        if (not read) and (line.split()[0] == "NODE_COORD_SECTION"):
            read = True
        elif (line.split()[0] == "EOF"):
            return coords
        elif read:
            data = line.split()
            coords.append(Coord(float(data[1]),float(data[2])))
    return "Error"

def main():
    if len(sys.argv) != 2:
        print("Error, 2 arguments are needed!")
    else:
        if (sys.argv[1] == "drill"):
            coords = read_file("a280.txt")
        elif (sys.argv[1] == "us"):
            coords = read_file("us48.txt")
        elif (sys.argv[1] == "burma"):
            coords = read_file("burma14.txt")
        else:
            print("Error, wrong arguments given!")
            return
        best_route=genetic_algorithm(coords,100)
        best_route_mem=genetic_algorithm(coords,100,True)
        plot_statistics(best_route,best_route_mem)

main()
