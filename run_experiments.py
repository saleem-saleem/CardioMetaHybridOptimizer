import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cmho_core import run_cmho
from data_preprocessing import load_and_preprocess_data

# --- Configuration Settings ---

# List of dataset paths (Update based on your folder structure)
DATASET_PATHS = [
    "./datasets/cleveland.csv",
    "./datasets/hungarian.csv",
    "./datasets/statlog.csv",
    "./datasets/switzerland.csv",
    "./datasets/long-beach-va.csv"
]

# List of classifiers to evaluate
CLASSIFIERS = ["svm", "rf", "xgb", "cnn-lstm"]

# Folder to save results
RESULTS_DIR = "./results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Helper Functions ---

def save_results(dataset_name, classifier_name, selected_features, performance):
    """Save selected features and classifier performance to a CSV file."""
    filename = f"{dataset_name}_{classifier_name}_results.csv"
    filepath = os.path.join(RESULTS_DIR, filename)

    # Build a dictionary of results
    results = {
        "Selected Features": [list(selected_features)],
        "Accuracy": [performance.get('accuracy', 0)],
        "Precision": [performance.get('precision', 0)],
        "Recall": [performance.get('recall', 0)],
        "F1-Score": [performance.get('f1_score', 0)]
    }

    # Save results as CSV
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"✅ Results saved to {filepath}")

def plot_summary_graph(all_results):
    """Plot a bar graph summarizing the accuracy of classifiers across datasets."""
    plt.figure(figsize=(12, 6))
    
    datasets = list(all_results.keys())
    x = np.arange(len(datasets))
    width = 0.2

    for idx, classifier in enumerate(CLASSIFIERS):
        accuracies = [all_results[dataset][classifier]['accuracy'] for dataset in datasets]
        plt.bar(x + idx * width, accuracies, width, label=classifier.upper())

    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy across Datasets using CMHO')
    plt.xticks(x + width * 1.5, datasets, rotation=30)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "summary_accuracy_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Summary plot saved to {plot_path}")

# --- Main Function ---

def main():
    """Run CMHO optimization on all datasets and classifiers."""
    all_results = {}  # To collect results for plotting later

    for dataset_path in DATASET_PATHS:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"\n🗂️  Dataset: {dataset_name}")
        
        # Load and preprocess dataset
        X, y = load_and_preprocess_data(dataset_path)
        
        all_results[dataset_name] = {}
        
        for classifier_name in CLASSIFIERS:
            print(f"⚙️ Running CMHO with Classifier: {classifier_name.upper()}")

            # Run CMHO
            selected_features, model, performance = run_cmho(X, y, classifier_name=classifier_name)
            
            # Save individual experiment results
            save_results(dataset_name, classifier_name, selected_features, performance)
            
            # Store for summary plotting
            all_results[dataset_name][classifier_name] = performance

    # After all experiments, plot a summary
    plot_summary_graph(all_results)

# --- Entry Point ---

if __name__ == "__main__":
    main()
