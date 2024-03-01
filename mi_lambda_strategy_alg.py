import math
import time
import numpy as np
import matplotlib.pyplot as plt

from cec2017.functions import *


class EvolutionaryAlgorithm_1:
    def __init__(self, max_iterations, lamb_rate, sigma):
        self.max_iterations = max_iterations
        self.sigma = sigma
        self.lamb_rate = lamb_rate
        self.fittest_individuals = []
        self.mean_sigma = []

    class Population:
        def __init__(self, individuals, grades, sigmas):
            self.individuals = individuals
            self.grades = grades
            self.sigmas = sigmas

        def __add__(self, other):
            return EvolutionaryAlgorithm_1.Population(
                np.vstack((self.individuals, other.individuals)),
                np.concatenate((self.grades, other.grades)),
                np.vstack((self.sigmas, other.sigmas)),
            )

    def find_fittest_individual(self, population):
        candidates = list(
            zip(population.individuals, population.grades, population.sigmas)
        )
        candidates.sort(key=lambda x: x[1])
        self.fittest_individuals.append(candidates[0])

    def generate_offspring(self, population, f):
        selected_indexes = [
            np.random.randint(0, len(population.grades) - 1) for _ in range(self.lamb)
        ]
        new_individuals = []
        new_grades = []
        new_sigmas = []
        n = len(population.individuals[0])
        tau = 1 / np.sqrt(2 * n)
        tau_prim = 1 / np.sqrt(2 * math.sqrt(n))
        for i in selected_indexes:
            a = np.random.normal(0, 1)
            b = np.random.normal(0, 1, n)
            new_sigma = population.sigmas[i] * np.exp(tau_prim * a + tau * b)
            new_individual = population.individuals[i] + new_sigma * np.random.normal(
                0, 1, n
            )
            new_individuals.append(new_individual)
            new_sigmas.append(new_sigma)
        new_grades.append(f(new_individuals))
        return self.Population(
            np.array(new_individuals),
            np.array(new_grades).flatten(),
            np.array(new_sigmas),
        )

    def select_new_generation(self, old_population, children):
        whole_population = old_population + children
        new_generation = sorted(
            zip(
                whole_population.individuals,
                whole_population.grades,
                whole_population.sigmas,
            ),
            key=lambda x: x[1],
            reverse=False,
        )[: self.mi]
        best_individual = new_generation[0]
        new_individuals, new_grades, new_sigmas = list(zip(*new_generation))
        self.mean_sigma.append(np.mean(np.array(new_sigmas).flatten()))
        return self.Population(new_individuals, new_grades, new_sigmas), best_individual

    def optim(self, f, initial_population):
        self.mi = len(initial_population)
        self.lamb = int(self.mi * self.lamb_rate)
        t = 0
        result = []
        initial_sigmas = np.full((self.mi, len(initial_population[0])), self.sigma)
        population_evaluation = self.Population(
            initial_population, np.array(f(initial_population)), initial_sigmas
        )
        self.find_fittest_individual(population_evaluation)
        while t < self.max_iterations:
            result.append(self.fittest_individuals[-1][1])
            children = self.generate_offspring(population_evaluation, f)
            population_evaluation, best_individual = self.select_new_generation(
                population_evaluation, children
            )
            if best_individual[1] < self.fittest_individuals[-1][1]:
                self.fittest_individuals.append(best_individual)
            t += 1
        return result, self.mean_sigma, self.fittest_individuals[-1]


def show_results_1(result, fittest_individual, sigma, mi, lambda_rate):
    print(fittest_individual)
    label = f"Outcome: {fittest_individual[1]}"
    plt.plot(range(len(result)), result, label=label)
    plt.title(
        f"Determined values for mi = {mi}, lambda = {int(mi*lambda_rate)}, initial sigma = {sigma}"
    )
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


def show_mean_sigma(mean_sigma):
    plt.plot(range(len(mean_sigma)), mean_sigma)
    plt.title("Mean sigma values")
    plt.xlabel("Iteration number")
    plt.ylabel("Mean sigma value")
    plt.show()


def test_sigma_values(x0):
    initial_sigma = [1000, 100, 10, 1, 0.1]
    for sigma in initial_sigma:
        ea = EvolutionaryAlgorithm_1(350, 0.7, sigma)
        result, mean_sigmas, fittest_individual = ea.optim(lambda x: f9(x), x0)
        plt.plot(range(len(result)), result, label=f"Init sigma: {sigma}")
    plt.title(f"Determined values for different initial sigma")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


def test_sigma_values_2(x0):
    initial_sigma = [1000, 100, 50, 10]
    for sigma in initial_sigma:
        ea = EvolutionaryAlgorithm_1(350, 0.7, sigma)
        result, mean_sigmas, fittest_individual = ea.optim(lambda x: f4(x), x0)
        plt.plot(range(len(mean_sigmas)), mean_sigmas)
        plt.title(f"Mean sigma values for initial sigma = {sigma}")
        plt.xlabel("Iteration number")
        plt.ylabel("Mean sigma value")
        plt.show()


def test_lambda_rates(x0):
    lambda_rates = [0.2, 0.5, 0.9, 1.2, 2]
    for lamb_rate in lambda_rates:
        ea = EvolutionaryAlgorithm_1(250, lamb_rate, 5)
        result, mean_sigmas, fittest_individual = ea.optim(lambda x: f9(x), x0)
        plt.plot(range(len(result)), result, label=f"Lambda: {int(lamb_rate*len(x0))}")
    plt.title(f"Determined values for different lambda values (mi = {len(x0)})")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    min_value = -1000
    max_value = 1000
    samples = 500
    dimension = 10
    x0 = np.random.uniform(min_value, max_value, size=(samples, dimension))

    lamb_rate = 0.8
    sigma = 50
    ea = EvolutionaryAlgorithm_1(500, lamb_rate, sigma)
    start_time = time.time()
    result, mean_sigmas, fittest_individual = ea.optim(lambda x: f9(x), x0)
    end_time = time.time()
    show_results_1(result, fittest_individual, sigma, len(x0), lamb_rate)
    show_mean_sigma(mean_sigmas)
    print(f"Time: {end_time- start_time}")

    test_lambda_rates(x0)

    test_sigma_values(x0)

    test_sigma_values_2(x0)
