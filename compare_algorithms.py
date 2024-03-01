from mi_lambda_strategy_alg import *
from regular_evolutionary_alg import *

if __name__ == "__main__":
    min_value = -100
    max_value = 100
    samples = 100
    dimension = 10
    x0 = np.random.uniform(min_value, max_value, size=(samples, dimension))

    lamb_rate = 0.8
    sigma = 50
    ea1 = EvolutionaryAlgorithm_1(2000, lamb_rate, sigma)
    start_time = time.time()
    result, mean_sigmas, fittest_individual = ea1.optim(lambda x: f9(x), x0)
    end_time = time.time()
    print(f"Time: {end_time - start_time}")

    label = f"Strategy: {fittest_individual[1]}"
    plt.plot(range(len(result)), result, label=label)

    ea2 = EvolutionaryAlgorithm_2(2000, 0.3, 0.8, 3, 0.8)
    start_time = time.time()
    results, fittest = ea2.optim(lambda x: f9(x), x0)
    end_time = time.time()
    print(f"Time: {end_time - start_time}")

    label = f"Standard: {fittest[1]}"
    plt.plot(range(len(results)), results, label=label)

    plt.title(f"Determined values for standard and strategy algorithms")
    plt.xlabel("Iteration number")
    plt.ylabel("Value of the target function")
    plt.yscale("log")
    plt.legend()
    plt.show()
