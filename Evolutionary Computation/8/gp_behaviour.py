import random
import math
import operator
import matplotlib.pyplot as plt
import numpy as np
from deap import base, gp, creator, tools, algorithms
import pygraphviz as pgv


data_points = [(-1, 0), (-0.9, -0.1629), (-0.8, -0.2624), (-0.7, -0.3129), (-0.6, -0.3264),
        (-0.5, -0.3125), (-0.4, -0.2784), (-0.3, -0.2289), (-0.2, -0.1664), (-0.1, -0.0909),
        (0, 0), (0.1, 0.1111), (0.2, 0.2496), (0.3, 0.4251), (0.4, 0.6496), (0.5, 0.9375),
        (0.6, 1.3056), (0.7, 1.7731), (0.8, 2.3616), (0.9, 3.0951), (1, 4)]


def protected_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else 1


def protected_log(arg):
    return np.log(abs(arg)) if arg != 0 else 0


def fitness_function(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the absolute errors between the expression and the real function
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
pset.renameArguments(ARG0="x")

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


def plot_statistics(best_individuals):
    number_of_generations = len(best_individuals)
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


def draw_solution(individual):
    nodes, edges, labels = gp.graph(individual)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("best_solution.pdf")


def additional_statistics():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def main():
    random.seed(167)

    population_size, crossover_rate, mutation_rate, number_of_generations = 1000, 0.7, 0, 50
    stats = additional_statistics()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    verbose = True

    # Initial population
    population = toolbox.population(n=population_size)
    # Evaluate the initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # best individuals of each generation are saved for statistics
    best_individuals = []
    for generation in range(1, number_of_generations+1):
        # Tournament selection of the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Apply crossover or mutation to offsprings
        offspring = algorithms.varOr(offspring, toolbox, population_size, crossover_rate, mutation_rate)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        # Find the best individual
        (best_fitness, best_ind) = (math.inf, None)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if fit[0] < best_fitness:
                (best_fitness, best_ind) = (fit[0], ind)

        best_individuals.append(best_ind)
        # Generational scheme: the population is entirely replaced by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    plot_statistics(best_individuals)

    final_solution = best_individuals[-1]
    draw_solution(final_solution)

    return final_solution


if __name__ == "__main__":
    main()