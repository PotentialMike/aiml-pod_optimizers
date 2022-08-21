import random
import numpy as np
import matplotlib.pyplot as plt

import lib.functions as fn

MAX_POPULATION = 40
CUTOFF = 5


class GeneticAlgorithmOptimizer:

    def __init__(self, test_function):

        # instance variables
        self.min = test_function.range_min
        self.max = test_function.range_max
        self.Function_name = test_function.name

        # instantiate test function class
        self.Function = getattr(fn, self.Function_name)()

    def generate_population(self, pop=MAX_POPULATION):
        pop = np.random.uniform(
            low=self.min, high=self.max, size=(pop, 2))
        self.evaluate_fitness(pop)
        fitness = self.evaluate_fitness(pop)
        self.population = np.column_stack((pop, fitness))

    def crossover(self, cutoff=CUTOFF, eval=False):
        x = self.population[0:cutoff, 0]
        y = self.population[0:cutoff, 1].copy()
        np.random.shuffle(y)
        self.offspring = np.column_stack((x, y))
        if eval:
            fitness = self.evaluate_fitness(self.offspring)
            self.offspring = np.column_stack((self.offspring, fitness))

    def mutate(self, radioactivity=0.1, eval=True):
        for child in range(self.offspring.shape[0]):
            for mutation_point in range(self.offspring.shape[1]):
                mutate_by = np.random.uniform(
                    self.min*radioactivity, self.max*radioactivity, 1)
                self.offspring[child, mutation_point] = \
                    self.offspring[child, mutation_point] + mutate_by
        if eval:
            fitness = self.evaluate_fitness(self.offspring)
            self.offspring = np.column_stack((self.offspring, fitness))

    def evaluate_fitness(self, x, y=None):
        # if `x` is 2-D, then `y` is unnecessary
        if y is None:
            x, y = np.split(x, 2, axis=1)
        return self.Function.eval(x, y)

    def sort_population(self):
        # sort on fitness (3rd column)
        self.population = self.population[self.population[:, 2].argsort()]

    def join_populations(self):
        self.population = np.append(self.population, self.offspring, axis=0)
        self.sort_population()

    def plot_population(self):

        # surface plot as base layer
        self.Function.plot_surface()

        # lay population on top
        plt.scatter(self.population[:, 0], self.population[:, 1],
                    marker="x", c='k')

        # lay population on top
        plt.scatter(self.offspring[:, 0], self.offspring[:, 1],
                    marker="+", c='r')


def main():

    # generate test data
    # func_list = [fn.Ackley, fn.Beale, fn.Easom, fn.Himmelblau,
    #  fn.HolderTable, fn.Rastrigin, fn.ShafferNo4]
    func_list = [fn.Ackley, fn.Beale, fn.Easom, fn.Himmelblau]
    test_function = random.choice(func_list)()
    test_function.plot_surface()

    # apply (minimum-finding) optimization algorithm
    opt = GeneticAlgorithmOptimizer(test_function)
    opt.generate_population()
    opt.Function.plot_surface()
    # plt.scatter(opt.population[:, 0], opt.population[:, 1],
    #             marker="x", c='y', alpha=0.8)

    # do-while loop; terminate
    rounds = 20
    for round in range(1, rounds):
        opt.sort_population()
        opt.crossover(eval=False)
        opt.mutate(radioactivity=0.5/round, eval=True)

        plt.scatter(opt.offspring[:, 0], opt.offspring[:, 1],
                    marker="+", c='r', alpha=0.5*round/rounds)

        opt.join_populations()

    print(opt.population[0:4, :])

    plt.show()
    # plt.show(block=False)
    # plt.pause(6)
    # plt.close()


if __name__ == "__main__":
    main()
