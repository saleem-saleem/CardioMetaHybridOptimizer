//Each optimization phase (LO, MPA, MRFO, Mutation, ILS) is its own function ✅

//Main CMHO loop handles fitness evaluation, phase switching, and convergence ✅

//Modular for easy updates later (you can tune parameters easily) ✅

//Final output: Selected features, trained model, and classification performance ✅




import numpy as np
from utils.fitness_functions import evaluate_fitness
from utils.mutation_operations import dynamic_mutation
from utils.evaluation_metrics import calculate_metrics
from classifiers import train_classifier
from data_preprocessing import load_and_preprocess_data

# --- Hyperparameters ---
POPULATION_SIZE = 50
MAX_ITERATIONS = 100
MUTATION_RATE = 0.2
SCALING_FACTOR = 0.7
INERTIA_WEIGHT = 0.8
CONVERGENCE_THRESHOLD = 0.01

# --- Core Functions ---

def global_exploration_LO(population):
    """Lemur Optimizer for global exploration."""
    leap_strength = 0.5
    for i in range(len(population)):
        random_leap = np.random.uniform(-leap_strength, leap_strength, size=population.shape[1])
        population[i] = np.clip(population[i] + random_leap, 0, 1)
    return np.round(population)

def intermediate_refinement_MPA(population, fitness):
    """Marine Predators Algorithm for intermediate refinement."""
    for i in range(len(population)):
        random_step = np.random.normal(0, 0.3, size=population.shape[1])
        population[i] = np.clip(population[i] + random_step * (fitness[i] / (np.max(fitness)+1e-9)), 0, 1)
    return np.round(population)

def focused_exploitation_MRFO(population):
    """Manta Ray Foraging Optimization for exploitation."""
    spiral_coeff = 2.0
    for i in range(len(population)):
        spiral_motion = np.sin(spiral_coeff * np.random.rand())
        population[i] = np.clip(population[i] + spiral_motion, 0, 1)
    return np.round(population)

def diversity_enhancement(population, fitness):
    """Apply dynamic mutation to enhance diversity."""
    return dynamic_mutation(population, fitness, mutation_rate=MUTATION_RATE)

def iterative_local_search_ILS(best_solution):
    """Local search around best solution."""
    neighborhood_size = 0.1
    perturbation = np.random.uniform(-neighborhood_size, neighborhood_size, size=best_solution.shape)
    new_solution = np.clip(best_solution + perturbation, 0, 1)
    return np.round(new_solution)

# --- Main Optimization Function ---

def run_cmho(X, y, classifier_name="cnn-lstm"):
    # Initialize population
    num_features = X.shape[1]
    population = np.random.randint(0, 2, size=(POPULATION_SIZE, num_features))
    
    best_solution = None
    best_fitness = -np.inf
    no_improvement_counter = 0

    for iteration in range(MAX_ITERATIONS):
        fitness = np.array([evaluate_fitness(X, y, individual, classifier_name) for individual in population])
        
        # Update best solution
        max_idx = np.argmax(fitness)
        if fitness[max_idx] > best_fitness:
            best_fitness = fitness[max_idx]
            best_solution = population[max_idx].copy()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Global Exploration
        population = global_exploration_LO(population)

        # Intermediate Refinement
        population = intermediate_refinement_MPA(population, fitness)

        # Focused Exploitation
        population = focused_exploitation_MRFO(population)

        # Diversity Maintenance
        if no_improvement_counter >= 5:
            population = diversity_enhancement(population, fitness)
            best_solution = iterative_local_search_ILS(best_solution)
            no_improvement_counter = 0

        # Convergence Check
        if best_fitness >= (1 - CONVERGENCE_THRESHOLD):
            print(f"Convergence achieved at iteration {iteration}")
            break

        print(f"Iteration {iteration}: Best Fitness = {best_fitness:.4f}")

    # Final evaluation
    selected_features = np.where(best_solution == 1)[0]
    X_selected = X[:, selected_features]
    model, performance = train_classifier(X_selected, y, classifier_name)

    return selected_features, model, performance

# --- Run if Main ---
if __name__ == "__main__":
    dataset_path = "./datasets/cleveland.csv"
    X, y = load_and_preprocess_data(dataset_path)
    
    selected_features, model, performance = run_cmho(X, y, classifier_name="cnn-lstm")
    
    print("\nSelected Features Indices:", selected_features)
    print("Performance Metrics:", performance)
