import numpy as np
from sklearn.model_selection import train_test_split
from classifiers import train_classifier

def evaluate_fitness(X, y, individual, classifier_name="cnn-lstm"):
    """Evaluate fitness based on selected features and classification performance."""
    selected_indices = np.where(individual == 1)[0]
    
    # If no feature selected, assign bad fitness
    if len(selected_indices) == 0:
        return 0
    
    X_selected = X[:, selected_indices]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    # Train classifier
    model, metrics = train_classifier(X_train, y_train, X_test, y_test, classifier_name)
    
    # Use accuracy (or F1-score) as fitness
    return metrics.get('accuracy', 0)
