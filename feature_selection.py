import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
import random

# Import your CMHO
from cmho_core import run_cmho

# --- 1. Recursive Feature Elimination (RFE) ---

def recursive_feature_elimination(X, y, num_features):
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=num_features)
    rfe.fit(X, y)
    selected = rfe.support_
    return np.where(selected)[0]

# --- 2. Genetic Algorithm (GA) Feature Selection ---

def genetic_algorithm_feature_selection(X, y, num_features, generations=20, population_size=30):
    num_features_total = X.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features_total)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalFitness(individual):
        if sum(individual) == 0:
            return 0,
        selected_features = [i for i, bit in enumerate(individual) if bit == 1]
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return accuracy,

    toolbox.register("evaluate", evalFitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    top_individual = tools.selBest(pop, k=1)[0]
    selected_features = np.where(np.array(top_individual) == 1)[0]
    
    # Select top N features
    if len(selected_features) > num_features:
        selected_features = selected_features[:num_features]
    
    return selected_features

# --- 3. Particle Swarm Optimization (PSO) Feature Selection (basic version) ---

def particle_swarm_optimization(X, y, num_features, iterations=20, swarm_size=30):
    num_features_total = X.shape[1]
    position = np.random.randint(0, 2, (swarm_size, num_features_total))
    velocity = np.random.uniform(-1, 1, (swarm_size, num_features_total))
    
    pbest = position.copy()
    pbest_fitness = np.zeros(swarm_size)

    for i in range(swarm_size):
        selected = np.where(position[i] == 1)[0]
        if len(selected) == 0:
            pbest_fitness[i] = 0
        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(X[:, selected], y)
            pbest_fitness[i] = model.score(X[:, selected], y)
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest = pbest[gbest_idx]
    
    w, c1, c2 = 0.5, 0.8, 0.9

    for iter in range(iterations):
        for i in range(swarm_size):
            velocity[i] = w*velocity[i] + c1*np.random.rand()*(pbest[i]-position[i]) + c2*np.random.rand()*(gbest-position[i])
            position[i] = 1 / (1 + np.exp(-velocity[i])) > 0.5

            selected = np.where(position[i] == 1)[0]
            if len(selected) == 0:
                fitness = 0
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X[:, selected], y)
                fitness = model.score(X[:, selected], y)

            if fitness > pbest_fitness[i]:
                pbest[i] = position[i]
                pbest_fitness[i] = fitness
        
        gbest_idx = np.argmax(pbest_fitness)
        gbest = pbest[gbest_idx]
    
    selected_features = np.where(gbest == 1)[0]
    if len(selected_features) > num_features:
        selected_features = selected_features[:num_features]
    return selected_features

# --- 4. Grey Wolf Optimizer (GWO) Feature Selection ---

def grey_wolf_optimizer(X, y, num_features, iterations=20, wolves=30):
    num_features_total = X.shape[1]
    wolves_pos = np.random.randint(0, 2, (wolves, num_features_total))
    
    for iter in range(iterations):
        fitness = []
        for wolf in wolves_pos:
            selected = np.where(wolf == 1)[0]
            if len(selected) == 0:
                fitness.append(0)
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X[:, selected], y)
                fitness.append(model.score(X[:, selected], y))
        
        sorted_indices = np.argsort(fitness)[::-1]
        alpha = wolves_pos[sorted_indices[0]]
        beta = wolves_pos[sorted_indices[1]]
        delta = wolves_pos[sorted_indices[2]]

        a = 2 - iter * (2 / iterations)
        
        for i in range(wolves):
            for j in range(num_features_total):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 2*a*r1 - a
                C1 = 2*r2
                D_alpha = abs(C1*alpha[j] - wolves_pos[i][j])
                X1 = alpha[j] - A1*D_alpha
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2*a*r1 - a
                C2 = 2*r2
                D_beta = abs(C2*beta[j] - wolves_pos[i][j])
                X2 = beta[j] - A2*D_beta
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2*a*r1 - a
                C3 = 2*r2
                D_delta = abs(C3*delta[j] - wolves_pos[i][j])
                X3 = delta[j] - A3*D_delta

                wolves_pos[i][j] = np.clip((X1+X2+X3)/3, 0, 1)
        wolves_pos = np.round(wolves_pos)
    
    alpha_selected = np.where(alpha == 1)[0]
    if len(alpha_selected) > num_features:
        alpha_selected = alpha_selected[:num_features]
    return alpha_selected

# --- 5. Lasso Regression Feature Selection ---

def lasso_feature_selection(X, y, num_features):
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X, y)
    model = SelectFromModel(lasso, max_features=num_features, prefit=True)
    selected = model.get_support()
    return np.where(selected)[0]

# --- 6. Feature Importance (Random Forest) ---

def feature_importance_selection(X, y, num_features):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return indices[:num_features]

# --- 7. Mutual Information Feature Selection ---

def mutual_info_selection(X, y, num_features):
    mi = mutual_info_classif(X, y)
    indices = np.argsort(mi)[::-1]
    return indices[:num_features]

# --- 8. CMHO Feature Selection ---

def cmho_feature_selection(X, y, classifier_name, num_features):
    selected_features, model, performance = run_cmho(X, y, classifier_name=classifier_name)
    if len(selected_features) > num_features:
        selected_features = selected_features[:num_features]
    return selected_features
