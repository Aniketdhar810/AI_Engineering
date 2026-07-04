# Machine Learning Study Notes

This repository contains study notes, notebooks, and reference materials for machine learning concepts.

## Directory Structure
* **[Supervised](file:///c:/Users/KIIT0001/Desktop/AI_Engineering/Machine_Learning/Supervised)**: Regression, classification, and evaluation metrics.
* **[Unsupervised](file:///c:/Users/KIIT0001/Desktop/AI_Engineering/Machine_Learning/Unsupervised)**: Clustering algorithms, dimensionality reduction, and spatial helpers.
* **[Evaluation](file:///c:/Users/KIIT0001/Desktop/AI_Engineering/Machine_Learning/Evaluation)**: Metric templates, model validation, and hyperparameter tuning notebooks.

---

## 1. Model Validation & Advanced Validation Techniques

Model validation is the process of checking how well a machine learning model generalizes to new, unseen data, preventing **overfitting** (where a model learns noise in training data instead of real patterns).

### Data Snooping
Data snooping occurs when training decisions or hyperparameter selections are influenced by looking at the test data too early. This leads to overly optimistic performance estimates that fail to translate to the real world.
> [!IMPORTANT]
> To avoid data snooping, keep the test set strictly isolated. Preprocessing steps (e.g., scaling, missing value imputation) should only fit on the training data and then transform the test data.

### Validation Strategies
To validate models correctly, partition data into three sets:
1. **Training Set ($70\%\text{--}80\%$)**: Used to train the parameters (weights) of the model.
2. **Validation Set**: Used during tuning to evaluate performance and choose hyperparameters.
3. **Test Set ($20\%\text{--}30\%$)**: Held back entirely and used only once at the end to estimate true generalization error.

```
+---------------------------------------+
|              All Data                 |
+-------------------+-------------------+
|      Train Set    |     Test Set      |
| (Train/Validation)|    (Final Evaluation)
+-------------------+-------------------+
```

### Cross-Validation
Instead of a single train/validation split, **Cross-Validation** uses data more efficiently:
* **K-Fold Cross-Validation**:
  1. Divide the dataset into $K$ equal-sized folds.
  2. For each hyperparameter setup, train on $K-1$ folds and validate on the remaining fold.
  3. Repeat this process $K$ times (each fold acts as the validation set once).
  4. Average the performance across all folds to pick the best hyperparameters.

* **Stratified K-Fold**:
  For classification with imbalanced target classes, stratified CV ensures each fold has the same class distribution as the original dataset, preventing biased evaluations.

* **Handling Skewed Targets**:
  In regression, skewed target distributions (e.g., long-tail house prices) can degrade model performance. Applying transformations like logarithmic $\log(y)$ or Box-Cox makes the distribution closer to a normal distribution, improving model stability and performance.

---

## 2. Supervised Learning: Classification Metrics

### The Confusion Matrix
A confusion matrix is a tabular layout that compares the true target class against the model's predicted class.

| | Predicted Positive | Predicted Negative |
| :--- | :---: | :---: |
| **Actual Positive** | True Positive (TP) | False Negative (FN) (Type II Error) |
| **Actual Negative** | False Positive (FP) (Type I Error) | True Negative (TN) |

### Key Metrics & Formulas

#### Accuracy
The ratio of correctly predicted instances to total instances.
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
> [!WARNING]
> Do not rely on accuracy for imbalanced datasets. A model predicting "not fraud" on a dataset with $99.9\%$ legitimate cases gets $99.9\%$ accuracy but misses all fraud.

#### Precision
The proportion of predicted positives that are actually positive.
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* **Use case**: Prioritize when **False Positives are costly** (e.g., sending a user's legitimate email to spam).

#### Recall (Sensitivity)
The proportion of actual positive instances correctly identified by the model.
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **Use case**: Prioritize when **False Negatives are costly** (e.g., failing to detect a critical disease in a patient).

#### F1 Score
The harmonic mean of precision and recall, balancing the trade-off between the two.
$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

---

## 3. Supervised Learning: Regression Metrics & Evaluation

Regression models predict continuous numerical values. Evaluating them requires measuring the size of prediction errors (residuals: $e_i = y_i - \hat{y}_i$).

### Key Metrics & Formulas

#### Mean Absolute Error (MAE)
The average of the absolute differences between predicted and actual values.
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **Properties**: Less sensitive to outliers; treats all errors linearly.

#### Mean Squared Error (MSE)
The average of the squared differences between predicted and actual values.
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Properties**: Strongly penalizes larger errors because errors are squared.

#### Root Mean Squared Error (RMSE)
The square root of the Mean Squared Error.
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Properties**: More interpretable because it retains the same units as the target variable ($y$).

#### R-squared ($R^2$ - Coefficient of Determination)
The proportion of variance in the target variable explained by the model's features.
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
* **Interpretation**:
  * $R^2 = 1$: Perfect predictions.
  * $R^2 = 0$: Performs no better than predicting the mean ($\bar{y}$).
  * $R^2 < 0$: Very poor performance (worse than predicting the mean).
  * *Note*: Can be misleading when applied to non-linear models.

#### Explained Variance
Measures the proportion of the target data's total variance captured by the model predictions.
$$\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$

### Visual Evaluation & Transformations
* **Visualization**: Plotting actual vs. predicted values in a scatter plot helps visually assess if errors are random (homoscedasticity) or exhibit patterns (heteroscedasticity).
* **Data Transformation**: Transforming the target variable using logarithmic or Box-Cox transformations concentrates skewed data, making relationships linear and improving metrics ($R^2 \uparrow$, error metrics $\downarrow$).

---

## 4. Regularization in Linear Regression

Regularization prevents overfitting by adding a penalty term to the regression loss function to shrink coefficients ($\beta_j$).

### Types of Regression

#### 1. Standard Linear Regression (OLS)
Minimizes standard Mean Squared Error with no penalties.
$$\text{Loss}_{\text{OLS}} = \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}\right)^2$$

#### 2. Ridge Regression (L2 Regularization)
Adds a penalty proportional to the sum of squared coefficients.
$$\text{Loss}_{\text{Ridge}} = \text{MSE} + \alpha \sum_{j=1}^{p} \beta_j^2$$
* **Behavior**: Shrinks coefficients towards zero, reducing sensitivity to noise and collinearity, but keeps all features (coefficients never become exactly zero).

#### 3. Lasso Regression (L1 Regularization)
Adds a penalty proportional to the sum of the absolute values of the coefficients.
$$\text{Loss}_{\text{Lasso}} = \text{MSE} + \alpha \sum_{j=1}^{p} |\beta_j|$$
* **Behavior**: Shrinks coefficients and drives some of them to exactly zero. This performs automatic **feature selection** by removing unimportant variables.

### Performance Scenarios

| Scenario | Best Choice | Rationale |
| :--- | :---: | :--- |
| **High Signal-to-Noise Ratio (Clear Data)** | All / Lasso | All models perform well; Lasso excels at isolating the true predictors. |
| **Low Signal-to-Noise Ratio (Noisy Data)** | Ridge / Lasso | Standard OLS overfits to noise; Ridge/Lasso stabilize predictions. |
| **Sparse Data (Few important features)** | Lasso | L1 penalty zeroes out the many redundant/unimportant features. |
| **Dense Data (Many important features)** | Ridge | L2 penalty works better when most features contribute slightly to the outcome. |

---

## 5. Evaluating Unsupervised Learning Models

Evaluating unsupervised learning is difficult because there are no true labels ("ground truth") to compare against. We evaluate models using structural statistics, stability, or external benchmarks.

### Clustering Evaluation Metrics

#### Internal Metrics (Unlabeled Data)
Evaluate clusters based only on the geometric features of the clusters themselves.
1. **Silhouette Score**: Measures how close a sample is to its own cluster (cohesion) compared to other clusters (separation). Ranges from $-1$ to $+1$.
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   *Where $a(i)$ is the mean distance to other points in the same cluster, and $b(i)$ is the mean distance to points in the nearest neighboring cluster.*
2. **Davies-Bouldin Index (DBI)**: Computes the average similarity ratio between each cluster and its most similar one. Lower values indicate tighter, better-separated clusters.
   $$\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{s_i + s_j}{d(c_i, c_j)} \right)$$
3. **Inertia (Within-Cluster Sum of Squares - WCSS)**: Measures the distance of data points to their nearest cluster centroid. Lower values mean tighter clusters.
   $$\text{Inertia} = \sum_{j=1}^{k} \sum_{i \in C_j} \|x_i - \mu_j\|^2$$

#### External Metrics (Labeled Data)
When true labels are temporarily available for benchmarking.
1. **Adjusted Rand Index (ARI)**: Compares cluster assignments against true labels, corrected for chance. Range: $[-1, 1]$ ($1$ is perfect).
2. **Normalized Mutual Information (NMI)**: Measures the shared information entropy between predictions and true labels. Range: $[0, 1]$.
3. **Fowlkes-Mallows Index (FMI)**: Combines pairwise precision and recall.
   $$\text{FMI} = \sqrt{\text{Precision} \times \text{Recall}} = \sqrt{\frac{\text{TP}}{\text{TP} + \text{FP}} \times \frac{\text{TP}}{\text{TP} + \text{FN}}}$$

#### Clustering Stability
Stability checks if a clustering algorithm returns similar groupings when the dataset is perturbed slightly (e.g., bootstrapping or adding noise). Stable configurations indicate robust patterns.

### Dimensionality Reduction Metrics
1. **Explained Variance Ratio**: (In PCA) Indicates the proportion of total data variance retained by each principal component.
2. **Reconstruction Error**: Measures the difference between the original high-dimensional data and the reconstructed data after projecting it back from the low-dimensional space.
3. **Neighborhood Preservation**: Measures how well points that were neighbors in the high-dimensional space remain neighbors in the low-dimensional space (crucial for t-SNE/UMAP).

---

## 6. Common Machine Learning Pitfalls

### Data Leakage
Data leakage occurs when information from the target variable or future test data accidentally leaks into the training pipeline.
* **Examples**: 
  * Calculating global statistics (e.g., dataset mean/std) before splitting into train/test sets.
  * Time-series predictions: predicting past events using future observations.
* **Mitigation**:
  * Build separate pipeline objects (using `scikit-learn`'s `Pipeline`).
  * For time-based data, split sequentially (train on past, test on future).

### Feature Importance & Confounding Pitfalls
1. **Correlation vs. Causation**: Feature importance metrics (like Random Forest Gini importance or Linear coefficients) show correlations, not direct causal relationships.
2. **Multicollinearity**: If features are highly correlated, their importances are shared or divided, making individual features look less important than they actually are.
3. **Class Imbalance**: Ignoring minority classes can skew feature importance toward predicting the majority class.
4. **Omitted Variables (Confounding)**: Failing to include a true causal variable can bias the coefficients of the included features (spurious relationships).
