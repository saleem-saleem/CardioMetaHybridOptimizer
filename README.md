# CardioMetaHybridOptimizer
CardioMetaHybridOptimizer (CMHO) is a multiphase hybrid algorithm for feature selection in cardiovascular disease prediction. It combines LO, MPA, MRFO, mutation, and ILS to boost accuracy, reduce error, and enhance convergence on medical datasets.

<p><b> Features </b></p>
<li> Hybrid optimization using LO, MPA, MRFO</li>

<li>Dynamic mutation and local search (ILS)</li>

<li>Supports classifiers: SVM, RF, XGBoost, CNN-LSTM</li>

<li>Benchmark-tested on 5 cardiovascular datasets</li>

<li>Improves accuracy, F1-score, and reduces error rates</li>

<br><p><b> Applications </b></p></br>
<li> CMHO is designed to enhance the accuracy of CVD diagnosis by selecting the most relevant clinical and lifestyle features. It supports early detection and risk classification to improve patient outcomes.</li>

<li>It can be applied to other health domains like diabetes, cancer, and Alzheimer’s disease. CMHO efficiently reduces high-dimensional data to essential variables, improving model performance.</li>

<li>The algorithm can be integrated into wearable or IoT-based health systems for real-time decision support. Its lightweight feature subsets make deployment faster and more efficient.</li>

<li>CMHO helps researchers identify key biomarkers and stratify patient groups in clinical datasets. This supports precision medicine and data-driven healthcare innovation.</li>

<li>By filtering out irrelevant or redundant features, CMHO enhances model training efficiency and generalization. It reduces overfitting and boosts performance across various AI tasks.</li>

<br><p><b> Prerequisites for Using the Framework </b></p></br>
<li> Python 3.8 or later</li>
<li>numpy==1.24.4 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1 seaborn==0.12.2 scipy==1.10.1 tensorflow==2.12.0 torch==2.0.0 deap==1.3.3</li>

<li>Input datasets should be in .csv format with preprocessed numerical/categorical values. Ensure missing values are imputed and categorical features are one-hot encoded.</li>

<li> 8 GB RAM, Intel i5/i7 or equivalent CPU. NVIDIA GTX 1650  is suggested for deep learning models like CNN-LSTM.</li>

<br><p><b>Workflow of CMHO (CardioMetaHybridOptimizer)</br></p></b>

Initialization
A population of candidate feature subsets is randomly generated as binary vectors. Each solution is evaluated using a fitness function based on classification performance (e.g., accuracy, F1-score).

Global Exploration (Lemur Optimizer - LO)
The population undergoes wide exploration using random leaping behavior. This phase ensures diverse coverage of the feature space to avoid premature convergence.

Intermediate Refinement (Marine Predators Algorithm - MPA)
High-potential solutions are adaptively refined based on cooperative hunting behavior.The search space is narrowed around the most promising feature subsets.

Focused Exploitation (Manta Ray Foraging Optimization - MRFO)
Solutions are fine-tuned using somersault foraging strategies.This phase performs intensive local exploitation to find optimal feature subsets.

Diversity Enhancement (Dynamic Mutation + Iterative Local Search - ILS)
If no improvement is observed, poorly performing solutions undergo fitness-ranked mutations.ILS explores neighborhoods around top solutions to recover from stagnation and enhance convergence.

Convergence Monitoring
Fitness improvements are continuously tracked.If the best solution stagnates or the maximum iterations are reached, the algorithm terminates.

Final Output
The best feature subset, along with classification metrics (accuracy, recall, precision, F1-score), is reported.







