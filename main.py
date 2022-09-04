import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd

import lib.functions as fn

MAX_POPULATION = 80
CUTOFF = 40
ROUNDS = 100
RADS = 1
VELOCITY = 0.1


class Optimizer:

    def __init__(self, test_function):

        # instance variables
        self.min = test_function.range_min
        self.max = test_function.range_max
        self.Function_name = test_function.name

        # instantiate test function class
        self.Function = getattr(fn, self.Function_name)()

        # core initial functions
        self.Function.plot_surface()

    def evaluate_fitness(self, x, y=None):
        # if `x` is 2-D, then `y` is unnecessary
        if y is None:
            x, y = np.split(x, 2, axis=1)
        return self.Function.eval(x, y)


class GeneticAlgorithm(Optimizer):

    def __init__(self, test_function):
        Optimizer.__init__(self, test_function)

        self.generate_population()

    def generate_population(self, pop=MAX_POPULATION):
        pop = np.random.uniform(
            low=self.min, high=self.max, size=(pop, 2))
        # TODO: is this line OBE? self.evaluate_fitness(pop)
        fitness = self.evaluate_fitness(pop)
        self.population = np.column_stack((pop, fitness))

    def optimize(self):
        for round in range(1, ROUNDS):
            self.sort_population()
            self.crossover(eval=False)
            self.mutate(radioactivity=RADS/round, eval=True)
            self.join_populations()

            # show optimization progress
            plt.scatter(self.offspring[:, 0], self.offspring[:, 1],
                        marker="+", c='r', alpha=0.5*round/ROUNDS)

        print(f"Bottom 5 Points:\n{self.population[0:5, :]}")

    def sort_population(self):
        # sort on fitness (3rd column)
        self.population = self.population[self.population[:, 2].argsort()]

    def crossover(self, cutoff=CUTOFF, eval=False):
        x = self.population[0:cutoff, 0]
        y = self.population[0:cutoff, 1].copy()
        np.random.shuffle(y)
        self.offspring = np.column_stack((x, y))

        if eval:
            fitness = self.evaluate_fitness(self.offspring)
            self.offspring = np.column_stack((self.offspring, fitness))

    def mutate(self, radioactivity, eval=True):
        for child in range(self.offspring.shape[0]):
            for mutation_point in range(self.offspring.shape[1]):
                mutate_by = np.random.uniform(
                    self.min*radioactivity, self.max*radioactivity, 1)
                self.offspring[child, mutation_point] = \
                    self.offspring[child, mutation_point] + mutate_by

        # clip any escaped mutations!
        self.offspring[self.offspring > self.max] = self.max
        self.offspring[self.offspring < self.min] = self.min

        if eval:
            fitness = self.evaluate_fitness(self.offspring)
            self.offspring = np.column_stack((self.offspring, fitness))

    def join_populations(self):
        self.population = np.append(self.population, self.offspring, axis=0)
        self.sort_population()


class ParticleSwarm(Optimizer):

    def __init__(self, test_function):
        Optimizer.__init__(self, test_function)

    def optimize(self):
        # initialize
        self.swarm_size = 20
        self.initial_positions()
        self.initial_velocities()

        # plt.scatter(self.positions[:, 0], self.positions[:, 1],
        #             marker="x", c='k')

        for round in range(1, ROUNDS):
            self.fitness = self.evaluate_fitness(self.positions)
            self.update_personal_bests()
            self.update_global_best()
            self.update_velocities()
            self.update_positions()

            plt.scatter(self.positions[:, 0], self.positions[:, 1],
                        marker="+", c='r', alpha=round/ROUNDS)

            plt.scatter(self.global_best_position[0], self.global_best_position[1],
                        marker="*", c='k')

            # plt.scatter(self.my_best_position[:, 0], self.my_best_position[:, 1],
            #             marker="*", c='b', alpha=round/ROUNDS)

    def update_positions(self):
        self.positions = self.positions + self.velocity

        # clip escapes
        self.positions[self.positions > self.max] = self.max
        self.positions[self.positions < self.min] = self.min

    def update_velocities(self):
        self.intertia = 0.5   # w, inertial weight
        self.cognitive = 0.1  # c1, cognitive coefficient
        self.social = 0.1     # c2, social coefficient
        r = np.random.rand(2)

        v1 = self.intertia * self.velocity
        v2 = self.cognitive * r[0] * (self.my_best_position - self.positions)
        v3 = self.social * r[1] * \
            (self.global_best_position - self.positions)

        self.velocity = v1 + v2 + v3

    def initial_positions(self):
        self.positions = np.random.uniform(
            low=self.min, high=self.max, size=(self.swarm_size, 2))

        # clip escapes
        self.positions[self.positions > self.max] = self.max
        self.positions[self.positions < self.min] = self.min

        self.my_best_position = self.positions.copy()
        self.my_best_fitness = self.evaluate_fitness(self.positions)
        self.global_best_fitness = self.my_best_fitness.min()
        self.global_best_position = self.positions[self.my_best_fitness.argmin(
        )]

    def initial_velocities(self):
        self.velocity = np.random.randn(self.swarm_size, 2) * VELOCITY

    def update_personal_bests(self):
        for index in range(self.swarm_size):
            if (self.fitness[index] <= self.my_best_fitness[index]):
                self.my_best_fitness[index] = self.fitness[index]
                self.my_best_position[index] = self.positions[index]

    def update_global_best(self):
        if (self.fitness.min() < self.global_best_fitness):
            self.global_best_fitness = self.fitness.min()
            self.global_best_position = \
                self.positions[self.fitness.argmin()]


def main():

    # spin the wheel for a test function!
    test_function = fn.picker()

    # apply an optimization algorithm
    # opt = GeneticAlgorithm(test_function)
    opt = ParticleSwarm(test_function)
    opt.optimize()

    # vizualize
    plt.show()
    # plt.show(block=False)
    # plt.pause(4)
    # plt.close()


if __name__ == "__main__":
    main()
