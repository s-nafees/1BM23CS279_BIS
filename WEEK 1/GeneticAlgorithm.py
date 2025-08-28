import numpy as np

# ==============================
# Portfolio Optimization with GA
# ==============================

# Example asset returns (mean returns) and covariance matrix
np.random.seed(42)
num_assets = 5
mean_returns = np.random.uniform(0.05, 0.2, num_assets)   # Expected returns
cov_matrix = np.random.rand(num_assets, num_assets)
cov_matrix = np.dot(cov_matrix, cov_matrix.T)              # Positive semi-definite
risk_free_rate = 0.0

# ------------------------------
# Genetic Algorithm Parameters
# ------------------------------
POP_SIZE = 50      # Population size
N_GEN = 100        # Number of generations
PC = 0.8           # Crossover probability
PM = 0.2           # Mutation probability

# ------------------------------
# Helper functions
# ------------------------------

def random_chromosome():
    """Generate a random portfolio (weights sum to 1)."""
    w = np.random.dirichlet(np.ones(num_assets))
    return w

def fitness_function(weights):
    """Sharpe ratio fitness function."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_vol == 0:
        return 0
    sharpe = (port_return - risk_free_rate) / port_vol
    return sharpe

def evaluate_fitness(population):
    return [fitness_function(ch) for ch in population]

def roulette_wheel_selection(population, fitness):
    total_fit = sum(fitness)
    pick = np.random.uniform(0, total_fit)
    current = 0
    for i, f in enumerate(fitness):
        current += f
        if current >= pick:
            return population[i]
    return population[-1]

def crossover(parent1, parent2):
    point = np.random.randint(1, num_assets-1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    # Normalize weights to sum to 1
    child1 /= np.sum(child1)
    child2 /= np.sum(child2)
    return child1, child2

def mutate(chromosome):
    pos = np.random.randint(0, num_assets)
    chromosome[pos] = np.random.random()
    chromosome /= np.sum(chromosome)
    return chromosome

# ------------------------------
# Genetic Algorithm Execution
# ------------------------------

def genetic_algorithm():
    # Step 1: Initialize population
    population = [random_chromosome() for _ in range(POP_SIZE)]

    for gen in range(N_GEN):
        fitness = evaluate_fitness(population)
        new_population = []

        while len(new_population) < POP_SIZE:
            # Selection
            p1 = roulette_wheel_selection(population, fitness)
            p2 = roulette_wheel_selection(population, fitness)

            # Crossover
            if np.random.rand() < PC:
                o1, o2 = crossover(p1.copy(), p2.copy())
            else:
                o1, o2 = p1.copy(), p2.copy()

            # Mutation
            if np.random.rand() < PM:
                o1 = mutate(o1)
            if np.random.rand() < PM:
                o2 = mutate(o2)

            new_population.append(o1)
            if len(new_population) < POP_SIZE:
                new_population.append(o2)

        population = new_population

    # Final evaluation
    fitness = evaluate_fitness(population)
    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx]

# ------------------------------
# Run GA
# ------------------------------

best_portfolio, best_fitness = genetic_algorithm()

print("Best Portfolio Weights:", best_portfolio)
print("Best Portfolio Sharpe Ratio:", best_fitness)
print("Expected Portfolio Return:", np.dot(best_portfolio, mean_returns))
print("Expected Portfolio Risk:", np.sqrt(np.dot(best_portfolio.T, np.dot(cov_matrix, best_portfolio))))


OUTPUT :
Best Portfolio Weights: [0.02614431 0.07650796 0.5324733  0.34949378 0.01538066]
Best Portfolio Sharpe Ratio: 0.18254777085442642
Expected Portfolio Return: 0.15258854251114842
Expected Portfolio Risk: 0.8358828037009056
