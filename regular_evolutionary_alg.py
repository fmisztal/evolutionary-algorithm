import random
import numpy as np
import time
import matplotlib.pyplot as plt

from cec2017.functions import *


class EvolutionaryAlgorithm_2:
    def __init__(
        self,
        max_iterations,
        mutation_rate,
        parent_ratio,
        tournament_size,
        mutation_strength,
        crossover=1,
    ):
        self.max_iterations = max_iterations
        self.mutation_strength = mutation_strength
        self.mutation_rate = mutation_rate
        self.parent_ratio = parent_ratio
        self.tournament_size = tournament_size
        self.fittest_individual = []
        self.crossover = crossover

    class Population:
        def __init__(self, individuals, grades):
            self.individuals = individuals
            self.grades = grades

    def find_fittest_individual(self, population):
        candidates = list(zip(population.individuals, population.grades))
        candidates.sort(key=lambda x: x[1])
        return candidates[0]

    def tournament_selection(self, population):
        selected_parents = []
        for _ in range(int(self.population_size * self.parent_ratio)):
            candidates = random.sample(
                list(zip(population.individuals, population.grades)),
                self.tournament_size,
            )
            candidates.sort(key=lambda x: x[1])
            selected_parents.append(candidates[0][0])
        return np.array(selected_parents)

    def single_point_crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            crossover_point = random.randint(1, len(parents[i]) - 1)
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1 = []
            child2 = []
            child1.extend(parent1[:crossover_point])
            child1.extend(parent2[crossover_point:])
            child2.extend(parent2[:crossover_point])
            child2.extend(parent1[crossover_point:])
            children.append(child1)
            children.append(child2)
        return np.array(children)

    def two_point_crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            cross_point_1 = random.randint(1, len(parents[i]) - 1)
            cross_point_2 = random.randint(1, len(parents[i]) - 1)
            while cross_point_1 == cross_point_2:
                cross_point_2 = random.randint(1, len(parents[i]) - 1)
            cross_point_1, cross_point_2 = sorted([cross_point_1, cross_point_2])
            child1 = []
            child2 = []
            child1.extend(parent1[:cross_point_1])
            child1.extend(parent2[cross_point_1:cross_point_2])
            child1.extend(parent1[cross_point_2:])
            child2.extend(parent2[:cross_point_1])
            child2.extend(parent1[cross_point_1:cross_point_2])
            child2.extend(parent2[cross_point_2:])
            children.append(child1)
            children.append(child2)
        return np.array(children)

    def mutate(self, children):
        mutated_children = children.copy()
        for individual in mutated_children:
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    individual[i] += self.mutation_strength * np.random.normal(0, 1)
        return np.array(mutated_children)

    def select_new_generation(self, old_population, children):
        k = len(old_population.grades) - len(children.grades)
        all_children = sorted(
            zip(children.individuals, children.grades),
            key=lambda x: x[1],
            reverse=False,
        )
        top_old_population = sorted(
            zip(old_population.individuals, old_population.grades),
            key=lambda x: x[1],
            reverse=False,
        )[:k]
        new_population = all_children + top_old_population
        individuals, grades = zip(*new_population)
        return self.Population(np.array(individuals), np.array(grades))

    def optim(self, f, initial_population):
        self.population_size = len(initial_population)
        t = 0
        result = []
        population_evaluation = self.Population(
            initial_population, f(initial_population)
        )
        self.fittest_individual.append(
            self.find_fittest_individual(population_evaluation)
        )
        while t < self.max_iterations:
            result.append(self.fittest_individual[-1][1])
            selected_parents = self.tournament_selection(population_evaluation)
            if self.crossover == 1:
                offspring = self.single_point_crossover(selected_parents)
            else:
                offspring = self.two_point_crossover(selected_parents)
            mutated_offspring = self.mutate(offspring)
            offspring_evaluation = self.Population(
                mutated_offspring, f(mutated_offspring)
            )
            fittest_child = self.find_fittest_individual(offspring_evaluation)
            if fittest_child[1] < self.fittest_individual[-1][1]:
                self.fittest_individual.append(fittest_child)
            population_evaluation = self.select_new_generation(
                population_evaluation, offspring_evaluation
            )
            t += 1
        return result, self.fittest_individual[-1]


def show_results_2(result, fittest_individual):
    print(fittest_individual)
    label = f"Outcome: {fittest_individual[1]}"
    plt.plot(range(len(result)), result, label=label)
    plt.title("Standard evolutionary algorithm")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


def test_parent_ratio_values(x0):
    parent_ratios = [0.3, 0.5, 0.7, 0.8, 1.0]
    for pr in parent_ratios:
        ea = EvolutionaryAlgorithm_2(2000, 0.3, pr, 3, 0.4)
        result, fittest_individual = ea.optim(lambda x: f4(x), x0)
        plt.plot(
            range(len(result)), result, label=f"Parent ratio: {pr} of the population"
        )
    plt.title(f"Determined values for different parent population sizes")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


def test_mutation_strength_values(x0):
    mutation_strengths = [0.3, 0.6, 0.8, 1.0, 2.0]
    for ms in mutation_strengths:
        ea = EvolutionaryAlgorithm_2(2000, 0.3, 0.8, 3, ms)
        result, fittest_individual = ea.optim(lambda x: f1(x), x0)
        plt.plot(range(len(result)), result, label=f"Mutation strength: {ms}")
    plt.title(f"Determined values for different parent mutation strengths")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


def test_mutation_rate_values(x0):
    mutation_rates = [0.3, 0.6, 0.8, 1.0]
    for mr in mutation_rates:
        ea = EvolutionaryAlgorithm_2(1500, mr, 0.8, 3, 0.6)
        result, fittest_individual = ea.optim(lambda x: f1(x), x0)
        plt.plot(range(len(result)), result, label=f"Mutation rate: {mr}")
    plt.title(f"Determined values for different parent mutation rates")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


def test_crossovers(x0):
    ea = EvolutionaryAlgorithm_2(2000, 0.3, 0.8, 3, 0.8, 1)
    result, fittest_individual = ea.optim(lambda x: f1(x), x0)
    plt.plot(range(len(result)), result, label=f"Single-point crossover")

    ea = EvolutionaryAlgorithm_2(2000, 0.3, 0.8, 3, 0.8, 2)
    result, fittest_individual = ea.optim(lambda x: f1(x), x0)
    plt.plot(range(len(result)), result, label=f"Two-point crossover")

    plt.title(f"Determined values for different crossover types")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    min_value = -1000
    max_value = 1000
    samples = 100
    dimension = 10
    x0 = np.random.uniform(min_value, max_value, size=(samples, dimension))

    ea = EvolutionaryAlgorithm_2(1000, 0.3, 0.7, 3, 20.0)
    start_time = time.time()
    result, fittest = ea.optim(lambda x: f1(x), x0)
    end_time = time.time()
    show_results_2(result, fittest)
    print(end_time - start_time)

    test_parent_ratio_values(x0)

    test_mutation_strength_values(x0)

    test_mutation_rate_values(x0)

    test_crossovers(x0)
