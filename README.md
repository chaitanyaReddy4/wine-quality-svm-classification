# Wine Quality Classification using Support Vector Machine (SVM)

A machine learning project that classifies red wine quality as **Good** or **Poor** using physicochemical properties and Support Vector Machine (SVM). This project demonstrates data preprocessing, feature engineering, feature scaling, model training, evaluation, and kernel comparison using Python and Scikit-learn.

## Project Overview

Wine quality is influenced by various physicochemical properties, including alcohol, acidity, sulphates, chlorides, and density. This project uses these properties to build a binary classification model.

### Target Definition

- **Poor (0):** Quality score less than 7
- **Good (1):** Quality score greater than or equal to 7

The original `quality` column is used to generate the target label and is excluded from the input features to prevent target leakage.

## Dataset

- **Dataset:** Red Wine Quality Dataset
- **File:** `winequality-red.csv`
- **Source:** [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Original records:** 1,599
- **Duplicate rows removed:** 240
- **Records after cleaning:** 1,359
- **Original input features:** 11
- **Engineered features:** 1 (`total_acidity`)
- **Total model input features:** 12

### Features Used

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Total acidity (engineered feature)

The `total_acidity` feature is calculated by adding fixed acidity, volatile acidity, and citric acid.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## Project Workflow

### 1. Data Collection and Exploration
- Load the red wine quality dataset.
- Inspect the dataset shape, columns, data types, and summary statistics.
- Check for missing values and duplicate records.

### 2. Data Preprocessing
- Remove 240 duplicate rows.
- Verify missing values.
- Convert the original quality score into binary classes.
- Analyze the distribution of Poor and Good wine samples.

### 3. Feature Engineering
Create the `total_acidity` feature:

`total_acidity = fixed acidity + volatile acidity + citric acid`

### 4. Correlation Analysis
Analyze relationships between numerical features and the binary target using a correlation matrix and heatmap.

The strongest observed correlations with the binary target were:

- Alcohol: +0.410
- Volatile acidity: -0.267
- Citric acid: +0.204
- Sulphates: +0.202
- Density: -0.158

Correlation indicates association, not causation.

### 5. Train-Test Split
- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Enabled

The final split contained 1,087 training samples and 272 test samples.

### 6. Feature Scaling
`StandardScaler` is used because SVM is sensitive to differences in feature magnitudes.

Median imputation and scaling are included in a Scikit-learn pipeline to prevent preprocessing statistics from leaking from the test set into training.

### 7. Model Training
The primary model uses `sklearn.svm.SVC` with the following configuration:

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1.0 |
| Gamma | `scale` |
| Class weight | `balanced` |
| Random state | 42 |

Balanced class weights help account for the unequal class distribution.

### 8. Model Evaluation
The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Five-fold stratified cross-validation

### 9. Feature Importance
Permutation importance is used to estimate how much the Good-class F1-score decreases when individual features are shuffled.

### 10. Kernel Comparison
Three SVM kernels are compared:

- Linear
- RBF
- Polynomial

All three use the same train-test split and preprocessing approach.

## Results

### RBF SVM — Test Set

| Metric | Result |
|---|---:|
| Accuracy | 79.78% |
| Precision (Good class) | 38.16% |
| Recall (Good class) | 78.38% |
| F1-score (Good class) | 51.33% |
| ROC-AUC | 0.886 |
| Mean five-fold CV accuracy | 80.68% |
| CV standard deviation | 0.0104 |

### Confusion Matrix

Rows represent actual classes and columns represent predicted classes. The class order is Poor (0), Good (1).

| Actual / Predicted | Poor (0) | Good (1) |
|---|---:|---:|
| Poor (0) | 188 | 47 |
| Good (1) | 8 | 29 |

- True Negatives: 188
- False Positives: 47
- False Negatives: 8
- True Positives: 29

The model identifies 29 of the 37 actual Good wine samples. However, 47 Poor samples are incorrectly classified as Good, which reduces precision.

### SVM Kernel Comparison

| Kernel | Accuracy | Precision (Good) | Recall (Good) | F1-score (Good) |
|---|---:|---:|---:|---:|
| Linear | 78.31% | 36.90% | 83.78% | 51.24% |
| RBF | 79.78% | 38.16% | 78.38% | 51.33% |
| Polynomial | 82.72% | 41.38% | 64.86% | 50.53% |

The Polynomial kernel achieved the highest test accuracy, while the RBF kernel achieved the highest F1-score for the Good class. The RBF kernel remains the primary model because it is specified in the project brief.

### Top Features by Permutation Importance

| Feature | Mean Importance |
|---|---:|
| Alcohol | 0.1003 |
| Sulphates | 0.0717 |
| Volatile acidity | 0.0264 |
| Density | 0.0133 |
| Chlorides | 0.0099 |

These values are specific to the trained model, test set, and F1-score evaluation. Negative permutation importance does not necessarily mean a feature is universally unimportant.

## Visualizations

The project includes the following visualizations:

- Wine quality class distribution
- Feature correlation heatmap
- Confusion matrix
- ROC curve
- Permutation feature importance
- SVM kernel comparison

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/wine-quality-svm-classification.git
cd wine-quality-svm-classification
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `winequality-red.csv` from the [UCI Wine Quality dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality).

Place the CSV in the project root, alongside the notebook.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Wine_Quality_SVM.ipynb` and run the cells in order, from top to bottom.

## Repository Structure

```text
wine-quality-svm-classification/
├── Wine_Quality_SVM.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── outputs/
    ├── correlation_heatmap.png
    ├── class_distribution.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── permutation_importance.png
    ├── kernel_comparison.png
    ├── evaluation_metrics.csv
    ├── kernel_comparison.csv
    ├── permutation_importance.csv
    └── predictions.csv
```

The output files listed above should be included only if they have actually been generated. The raw dataset can be downloaded separately and does not need to be committed to the repository.

## Key Findings

- Alcohol had the highest permutation importance in the reported RBF model.
- Sulphates and volatile acidity were also influential features.
- The RBF model achieved a ROC-AUC of 0.886.
- The model achieved 78.38% recall for the Good class but only 38.16% precision.
- The Polynomial kernel achieved the highest accuracy among the three tested kernels.
- The class imbalance makes precision, recall, and F1-score important alongside accuracy.

## Limitations and Future Improvements

- The dataset is imbalanced under the selected target definition: 184 Good samples and 1,175 Poor samples.
- The Good-class precision indicates that the model produces a substantial number of false-positive predictions.
- The engineered `total_acidity` feature may contain redundant information.
- Future improvements could include cross-validated hyperparameter tuning, careful feature selection, and comparison with other classification algorithms.
- Model selection should use training data and cross-validation, keeping the test set reserved for final evaluation.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for red wine quality classification using Support Vector Machines. It includes data cleaning, target creation, feature engineering, correlation analysis, stratified data splitting, feature scaling, model training, evaluation, cross-validation, permutation importance, and kernel comparison.

The RBF SVM achieved 79.78% test accuracy and a ROC-AUC of 0.886 in the reported run. The Polynomial kernel achieved higher accuracy, while the RBF model achieved a slightly higher Good-class F1-score. The results highlight the importance of evaluating imbalanced classification models using precision, recall, F1-score, and ROC-AUC rather than accuracy alone.

## Dataset Attribution

The dataset is provided by the UCI Machine Learning Repository. Review the dataset page and its terms before redistributing the raw data. If you intend to license your code, add a license file to the repository and ensure it does not misrepresent the dataset's terms.
