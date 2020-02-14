import random
import math
import operator
import matplotlib.pyplot as plt
import numpy as np
from deap import base, gp, creator, tools, algorithms

data_points = [(-1, 0), (-0.9, -0.1629), (-0.8, -0.2624), (-0.7, -0.3129), (-0.6, -0.3264),
        (-0.5, -0.3125), (-0.4, -0.2784), (-0.3, -0.2289), (-0.2, -0.1664), (-0.1, -0.0909),
        (0, 0), (0.1, 0.1111), (0.2, 0.2496), (0.3, 0.4251), (0.4, 0.6496), (0.5, 0.9375),
        (0.6, 1.3056), (0.7, 1.7731), (0.8, 2.3616), (0.9, 3.0951), (1, 4)]


def protected_div(numerator, denominator):
    if denominator != 0:
        return numerator / denominator
    else:
        return 1


def protected_log(arg):
    if arg != 0:
        return np.log(abs(arg))
    else:
        return 0


def fitness_function(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function
    abs_errors = [abs(func(x) - y) for x, y in points]
    sum_of_errors = math.fsum(abs_errors)
    return sum_of_errors,


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", fitness_function, points=data_points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(167)

    population_size = 1000
    crossover_rate = 0.7
    mutation_rate = 0
    number_of_generations = 50

    pop = toolbox.population(n=population_size)

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    best_individuals = []
    for g in range(number_of_generations):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, crossover_rate, mutation_rate)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        (best_fitness, best_ind) = (math.inf, None)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if fit[0] < best_fitness:
                (best_fitness, best_ind) = (fit[0], ind)

        best_individuals.append(best_ind)
        # The population is entirely replaced by the offspring
        pop[:] = offspring

    best_fitnesses = [ind.fitness.values for ind in best_individuals]
    best_sizes = [len(ind) for ind in best_individuals]

    # plot the results
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(range(1, number_of_generations + 1), best_fitnesses, "b-", label="Best Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(range(1, number_of_generations + 1), best_sizes, "r-", label="Best Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()

    final = best_individuals[-1]
    nodes, edges, labels = gp.graph(final)

    return final


if __name__ == "__main__":
    main()