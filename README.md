# Brain Tumor Machine Learning: Classification, Regression, and Clustering

## 1. Project Overview

This machine learning capstone project explores numerical features derived from brain tumor images using supervised classification, regression, and unsupervised clustering. It compares multiple machine learning algorithms, evaluates their performance using appropriate metrics, and presents results through visualizations and an interactive Streamlit dashboard.

## 2. Problem Statement

The project applies machine learning techniques to brain tumor image-derived data to:

- **Classification:** Classify samples into the available tumor categories.
- **Regression:** Predict continuous target values and compare regression algorithms using appropriate evaluation metrics.
- **Clustering:** Explore patterns and group structures without using tumor labels during model fitting.
- **Model evaluation:** Compare algorithms, perform hyperparameter tuning where applicable, and assess performance using suitable metrics and visualizations.

## 3. Dataset Description

The dataset used in the current notebooks is `segmentation_features.csv`.

The dataset contains image-derived numerical features such as:
- Tumor area and percentage
- Tumor width and height
- Aspect ratio
- Mean, standard deviation, minimum, and maximum intensity

The dataset also contains:
- `image_name`: sample/image identifier
- `tumor_type`: ground-truth tumor category

The clustering workflow excludes `image_name` and `tumor_type` from the clustering features. Ground-truth tumor labels are used only after clustering for interpretation.

**Dataset location:** The feature dataset is stored at `generated_data/segmentation_features.csv`, relative to the project root. Ensure that the notebooks and Streamlit application load the dataset from this location.

## 4. Project Structure

```text
ML PROJECT/
├── app.py
├── DATASET/
│   ├── classification/
│   │   ├── Testing/
│   │   │   ├── glioma/
│   │   │   ├── meninglioma/
│   │   │   ├── pitutary/
│   │   │   └── no tumor/
│   │   └── Training/
│   │       ├── glioma/
│   │       ├── meninglioma/
│   │       ├── pitutary/
│   │       └── no tumor/
│   └── segmentation/
│       ├── Glioma/
│       ├── Meninglioma/
│       └── Pitutary tumor/
├── generated_data/
│   ├── logistic_test_predictions.csv
│   ├── polynomial_test_predictions.csv
│   └── segmentation_features.csv
├── models/
│   ├── final_logistic_regression_model.pkl
│   ├── classification_scaler.pkl
│   ├── final_polynomial_regression_model.pkl
│   └── regression_scaler.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_data_generator.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_prediction_visualization.ipynb
│   ├── Classification.ipynb
│   ├── Regression.ipynb
│   └── Clustering.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```
This is a representative structure based on the current project folders. Keep any additional existing project folders and files.

## 5. Methods

### Classification

The classification notebook evaluates ten algorithms:
1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Classifier (SVC)
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. Bagging with a Decision Tree base estimator
10. Multi-Layer Perceptron (MLP)

Evaluation includes accuracy, weighted precision, weighted recall, weighted F1-score, confusion matrices, and multiclass ROC-AUC using a One-vs-Rest approach. Hyperparameter tuning and final model selection are documented in the notebook.

### Regression

The regression notebook evaluates ten algorithms, including linear, regularized, polynomial, tree-based, ensemble, support vector, and K-nearest-neighbors regression models.

Evaluation includes R², Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). The workflow also includes hyperparameter tuning, actual-versus-predicted plots, residual analysis, and feature-importance analysis for a tree-based model.

### Clustering

The clustering notebook evaluates:
- K-Means clustering, with an Elbow curve to help select the number of clusters.
- Agglomerative Hierarchical Clustering, with a dendrogram and linkage-strategy comparison.

Internal clustering metrics:
- Silhouette Score — higher is generally better.
- Davies–Bouldin Index — lower is generally better.
- Calinski–Harabasz Index — higher is generally better.

Visualizations include PCA plots for both clustering algorithms and a t-SNE plot for K-Means. Ground-truth labels are used only for post-clustering interpretation.

## 6. Results Summary

### Clustering results

The current four-cluster comparison produced the following results:

| Algorithm | Silhouette Score | Davies–Bouldin Index | Calinski–Harabasz Index |
|---|---:|---:|---:|
| K-Means | 0.2468 | 1.2790 | 798.4476 |
| Agglomerative (Ward) | 0.2156 | 1.3366 | 693.3752 |

In this comparison, K-Means scored better on all three internal metrics.

### Agglomerative linkage comparison

| Linkage | Silhouette Score | Davies–Bouldin Index | Calinski–Harabasz Index |
|---|---:|---:|---:|
| Average | 0.4279 | 0.7601 | 55.9395 |
| Ward | 0.2156 | 1.3366 | 693.3752 |
| Complete | 0.1982 | 1.3131 | 382.1469 |

Average linkage performed best on Silhouette and Davies–Bouldin, while Ward linkage performed best on Calinski–Harabasz. The preferred linkage depends on the evaluation criterion.

### Classification Results — Baseline Validation Metrics

Ten classification algorithms were evaluated using Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, and multiclass ROC-AUC (One-vs-Rest).

| Model | Accuracy | Weighted F1-Score | ROC-AUC (OvR) |
|---|---:|---:|---:|
| Logistic Regression | 0.6505 | 0.6057 | 0.8248 |
| KNN | 0.6474 | 0.6473 | 0.8094 |
| Gaussian Naive Bayes | 0.6049 | 0.5877 | 0.7876 |
| Decision Tree | 0.6413 | 0.6435 | 0.7336 |
| SVC | 0.6383 | 0.6157 | 0.8216 |
| Random Forest | 0.7021 | 0.6991 | 0.8565 |
| AdaBoost | 0.6170 | 0.5773 | 0.7928 |
| Gradient Boosting | 0.6687 | 0.6593 | 0.8390 |
| Bagging | 0.6869 | 0.6788 | 0.8562 |
| MLP | 0.6778 | 0.6569 | 0.8398 |

Random Forest achieved the highest baseline validation accuracy (0.7021) and weighted F1-score (0.6991) among the ten classifiers. Hyperparameter tuning produced a small weighted F1-score improvement, from 0.699126 to 0.699371, while validation accuracy remained approximately 0.7021. Final model performance should be reported separately using the held-out test set.

### Regression Results (Test Set)

Ten regression algorithms were evaluated on the held-out test set using R², RMSE, and MAE.

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Gradient Boosting Regressor | 0.9436 | 798.0817 | 473.0471 |
| Polynomial Regression | 0.9418 | 810.9453 | 509.4726 |
| Random Forest Regressor | 0.9381 | 836.0042 | 478.4498 |
| KNN Regressor | 0.9309 | 883.3956 | 594.8669 |
| Linear Regression | 0.9229 | 932.9473 | 647.7204 |
| Ridge Regression | 0.9229 | 933.2580 | 647.6691 |
| Lasso Regression | 0.9228 | 933.8718 | 647.9185 |
| Decision Tree Regressor | 0.8711 | 1206.6198 | 671.1763 |
| ElasticNet Regressor | 0.8661 | 1229.5609 | 838.6887 |
| SVR | 0.8132 | 1452.4051 | 722.2051 |

Gradient Boosting achieved the highest held-out test R² (0.9436), the lowest test RMSE (798.0817), and the lowest test MAE (473.0471) among the ten evaluated regression models.

Five-fold cross-validation was performed for Gradient Boosting Regressor and Polynomial Regression, achieving mean R² scores of 0.9508 and 0.9481, respectively. Both models had a standard deviation of 0.009 across the five folds.


## 7. Environment Setup

Use a Python environment compatible with the packages listed in `requirements.txt`.

From the project root, create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open the `notebooks` folder and run the required notebooks from top to bottom.

## 8. How to Run

1. Ensure the dataset is available in the documented dataset location.
2. Install dependencies from `requirements.txt`.
3. Open `notebooks/Classification.ipynb` to review supervised classification.
4. Open `notebooks/Regression.ipynb` to review the regression workflow.
5. Open `notebooks/Clustering.ipynb` to review unsupervised clustering.
6. Run all cells in each notebook in order and confirm there are no errors.
7. Save notebooks with outputs visible.

### Launch the interactive dashboard

From the project root, activate your virtual environment and run:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Confirm that the Classification, Regression, and Clustering pages load and that their predictions or evaluation outputs work as expected.

## 9. Reproducibility and Notes

- Random seeds are set where applicable in the notebooks.
- Clustering uses standardized numerical features.
- Ground-truth labels are excluded from clustering model fitting.
- Dataset paths may need adjustment when running on another machine.
- Reported results should be regenerated if preprocessing, data, or model settings change.

## 10. Repository

This project is version-controlled with Git. Commit meaningful changes separately, such as notebook updates, documentation, and dependency changes. Avoid committing temporary files, notebook checkpoints, or large raw datasets if they are excluded by `.gitignore`.
