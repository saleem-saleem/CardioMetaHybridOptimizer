import numpy as np

def dynamic_mutation(population, fitness, mutation_rate=0.2):
    """Apply fitness-ranked dynamic mutation."""
    worst_indices = np.argsort(fitness)[:len(population)//2]  # Bottom half
    
    for idx in worst_indices:
        for gene_idx in range(population.shape[1]):
            if np.random.rand() < mutation_rate:
                population[idx, gene_idx] = 1 - population[idx, gene_idx]  # Flip bit
    return population
