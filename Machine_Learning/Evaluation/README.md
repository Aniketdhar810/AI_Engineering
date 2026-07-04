# Evaluating and Validating Machine Learning Models

This directory contains resources, notebooks, and reference materials for evaluating, validating, and tuning machine learning models across classification, regression, and clustering tasks.

---

## Model Evaluation Metrics and Methods

### Classification Metrics

#### 1. `classification_report`
* **Description**: Generates a text report showing the main classification metrics (Precision, Recall, F1-score, and Support) for each class.
* **Key Parameters**:
  * `target_names`: Optional list of display names matching the class labels.
* **Pros**: Provides a quick, comprehensive, and clean summary of classification performance.
* **Limitations**: Might not provide enough depth or specific diagnostic details for highly imbalanced datasets.
* **Code Syntax**:
  ```python
  from sklearn.metrics import classification_report

  # y_true: True labels | y_pred: Predicted labels
  report = classification_report(y_true, y_pred, target_names=["class1", "class2"])
  print(report)
  ```

#### 2. `confusion_matrix`
* **Description**: Computes a confusion matrix to evaluate the classification accuracy of a model by detailing True/False Positives and Negatives.
* **Key Parameters**:
  * `labels`: List of labels to index the matrix.
* **Pros**: Crucial for understanding specific misclassifications (e.g., which class is being mistaken for another).
* **Limitations**: Does not provide information on prediction probabilities or confidence.
* **Code Syntax**:
  ```python
  from sklearn.metrics import confusion_matrix

  # y_true: True labels | y_pred: Predicted labels
  conf_matrix = confusion_matrix(y_true, y_pred)
  ```

---

### Regression Metrics

#### 3. `mean_squared_error` (MSE)
* **Description**: Calculates the average of the squared errors (differences between true and predicted values).
* **Key Parameters**:
  * `sample_weight`: Optional array of weights to apply to individual samples.
* **Pros**: Simple, differentiable, and widely used mathematically.
* **Limitations**: Extremely sensitive to outliers because errors are squared.
* **Code Syntax**:
  ```python
  from sklearn.metrics import mean_squared_error

  # y_true: True values | y_pred: Predicted values
  mse = mean_squared_error(y_true, y_pred)
  ```

#### 4. `root_mean_squared_error` (RMSE)
* **Description**: Computes the square root of the Mean Squared Error, returning the error metric to the original units of the target variable.
* **Key Parameters**:
  * `sample_weight`: Optional weights for samples.
* **Pros**: Highly interpretable since it shares the same scale/unit as the target variable.
* **Limitations**: Like MSE, it remains sensitive to large errors and outliers.
* **Code Syntax**:
  ```python
  from sklearn.metrics import root_mean_squared_error

  # y_true: True values | y_pred: Predicted values
  rmse = root_mean_squared_error(y_true, y_pred)
  ```

#### 5. `mean_absolute_error` (MAE)
* **Description**: Measures the average absolute magnitude of the errors without considering their direction.
* **Key Parameters**:
  * `sample_weight`: Optional weights for samples.
* **Pros**: Less sensitive to outliers compared to MSE or RMSE (errors are not squared).
* **Limitations**: Does not penalize large errors heavily.
* **Code Syntax**:
  ```python
  from sklearn.metrics import mean_absolute_error

  # y_true: True values | y_pred: Predicted values
  mae = mean_absolute_error(y_true, y_pred)
  ```

#### 6. `r2_score` (Coefficient of Determination)
* **Description**: Computes the proportion of variance in the dependent variable that is predictable from the independent variables.
* **Pros**: Standard metric offering a quick indicator of the goodness of fit ($1.0$ is a perfect fit).
* **Limitations**: Can be misleading for non-linear models.
* **Code Syntax**:
  ```python
  from sklearn.metrics import r2_score

  # y_true: True values | y_pred: Predicted values
  r2 = r2_score(y_true, y_pred)
  ```

#### 7. `explained_variance_score`
* **Description**: Computes the explained variance regression score, measuring the proportion of variance explained by the model's predictions.
* **Pros**: Complements $R^2$ by measuring how well the model captures the variability of the target.
* **Limitations**: Not applicable to classification tasks.
* **Code Syntax**:
  ```python
  from sklearn.metrics import explained_variance_score

  # y_true: True values | y_pred: Predicted values
  ev_score = explained_variance_score(y_true, y_pred)
  ```

---

### Clustering Validation & Spatial Metrics

#### 8. `silhouette_score`
* **Description**: Calculates the mean Silhouette Coefficient of all samples. It measures how similar an object is to its own cluster compared to other clusters.
* **Key Parameters**:
  * `metric`: The distance metric to use (e.g., `'euclidean'`).
* **Pros**: Useful for determining cluster cohesion and separation without knowing true labels.
* **Limitations**: Sensitive to outliers and computationally expensive on large datasets.
* **Code Syntax**:
  ```python
  from sklearn.metrics import silhouette_score

  # X: Data array | labels: Cluster assignments
  score = silhouette_score(X, labels, metric='euclidean')
  ```

#### 9. `silhouette_samples`
* **Description**: Computes the silhouette coefficient for each individual sample, facilitating visualization of cluster density/silhouette plots.
* **Key Parameters**:
  * `metric`: The distance metric to use.
* **Pros**: Provides granular, sample-level insights into cluster membership quality.
* **Limitations**: High computational overhead for large datasets.
* **Code Syntax**:
  ```python
  from sklearn.metrics import silhouette_samples

  # X: Data array | labels: Cluster assignments
  samples = silhouette_samples(X, labels, metric='euclidean')
  ```

#### 10. `davies_bouldin_score`
* **Description**: Computes the Davies-Bouldin index, which is defined as the average similarity measure of each cluster with its most similar cluster. Lower values signify better clustering.
* **Pros**: Simple to calculate and effective for comparing different clustering models.
* **Limitations**: Tends to favor spherical clusters and can behave poorly with highly imbalanced cluster sizes.
* **Code Syntax**:
  ```python
  from sklearn.metrics import davies_bouldin_score

  # X: Data array | labels: Cluster assignments
  db_score = davies_bouldin_score(X, labels)
  ```

#### 11. `Voronoi`
* **Description**: Generates a Voronoi tessellation/diagram, partitioning a 2D space into regions based on distance to specified generator points (centroids).
* **Pros**: Great for spatial boundary analysis and understanding distance partitioning.
* **Limitations**: Primarily useful in low-dimensional (2D/3D) spatial contexts.
* **Code Syntax**:
  ```python
  from scipy.spatial import Voronoi

  # points: Coordinates of generator points (centroids)
  vor = Voronoi(points)
  ```

#### 12. `voronoi_plot_2d`
* **Description**: Visualizes a 2D Voronoi diagram using Matplotlib.
* **Key Parameters**:
  * `show_vertices`: Set to `True` to display the vertices of the Voronoi regions.
* **Pros**: Excellent for intuitive visual exploration of spatial clustering boundaries.
* **Limitations**: Limited to 2D projections.
* **Code Syntax**:
  ```python
  from scipy.spatial import voronoi_plot_2d

  # vor: Voronoi object
  fig = voronoi_plot_2d(vor, show_vertices=True)
  ```

#### 13. `matplotlib.patches.Patch`
* **Description**: Creates custom graphical shapes (rectangles, circles, polygons) to annotate and customize plots (e.g., custom legends, boundary drawings).
* **Key Parameters**:
  * `color`: Fill color of the patch.
* **Pros**: Allows complete freedom in visual legend and area labeling.
* **Code Syntax**:
  ```python
  import matplotlib.patches as patches

  # Create a rectangle patch
  rectangle = patches.Rectangle((0, 0), 1, 1, color='blue')
  ```

---

### Regularization & Workflow Optimization

#### 14. Ridge Regression (L2 Regularization)
* **Description**: Linear regression with an L2 penalty on coefficients to prevent overfitting by shrinking them towards zero.
* **Key Parameters**:
  * `alpha`: Regularization strength (higher values indicate stronger regularization).
* **Pros**: Reduces model variance and addresses multicollinearity.
* **Limitations**: Does not perform feature selection (coefficients are shrunk close to zero, but never exactly zero).
* **Code Syntax**:
  ```python
  from sklearn.linear_model import Ridge

  ridge = Ridge(alpha=1.0)
  ```

#### 15. Lasso Regression (L1 Regularization)
* **Description**: Linear regression with an L1 penalty on coefficients, encouraging sparsity (some coefficients become exactly zero).
* **Key Parameters**:
  * `alpha`: Regularization strength.
* **Pros**: Built-in feature selection by driving irrelevant feature coefficients to zero.
* **Limitations**: Struggles when features are highly correlated (tends to pick one arbitrarily).
* **Code Syntax**:
  ```python
  from sklearn.linear_model import Lasso

  lasso = Lasso(alpha=0.1)
  ```

#### 16. `Pipeline`
* **Description**: Chains multiple preprocessing transformers and a final estimator into a single, cohesive unit.
* **Pros**: 
  * Eliminates data leakage during cross-validation.
  * Simplifies deployment and guarantees execution order.
* **Code Syntax**:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler

  pipeline = Pipeline(steps=[
      ('scaler', StandardScaler()), 
      ('model', Ridge(alpha=1.0))
  ])
  ```

#### 17. `GridSearchCV`
* **Description**: Performs an exhaustive search over a specified parameter grid, evaluated via cross-validation, to find the best model configuration.
* **Key Parameters**:
  * `param_grid`: Dictionary of parameter values to search over.
* **Pros**: Automatically identifies the optimal combination of hyperparameters.
* **Limitations**: Computationally expensive and slow for large parameter grids or datasets.
* **Code Syntax**:
  ```python
  from sklearn.model_selection import GridSearchCV

  grid_search = GridSearchCV(estimator=Ridge(), param_grid={'alpha': [0.1, 1.0, 10.0]})
  ```

---

## Visualization Strategies for K-Means Evaluation

Evaluating K-Means is challenging because there are no true labels. We use several diagnostic plotting techniques to choose the optimal number of clusters ($k$) and check the stability of our clustering solution.

### 1. Multiple Runs of K-Means
* **Description**: Executes the K-Means algorithm multiple times with different random seed initializations to check the stability of cluster assignments and centroid locations.
* **Advantage**: Helps check if the algorithm gets stuck in local minima (shows consistency/variability across runs).
* **Limitation**: High computational overhead for larger datasets.
* **Code Snippet**:
  ```python
  import matplotlib.pyplot as plt
  from sklearn.cluster import KMeans

  n_runs = 4
  inertia_values = []
  plt.figure(figsize=(12, 12))

  for i in range(n_runs):
      # Run K-Means with different random state initialization
      kmeans = KMeans(n_clusters=4, random_state=None)
      kmeans.fit(X)
      inertia_values.append(kmeans.inertia_)

      # Plot cluster results
      plt.subplot(2, 2, i + 1)
      plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
      plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x', label='Centroids')
      plt.title(f'K-Means Clustering Run {i + 1}')
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.legend()

  plt.tight_layout()
  plt.show()

  for i, inertia in enumerate(inertia_values, start=1):
      print(f'Run {i}: Inertia={inertia:.2f}')
  ```

### 2. Elbow Method (Inertia vs. $k$)
* **Description**: Plots the model inertia (within-cluster sum of squares) against different values of $k$. The "elbow" point indicates where adding more clusters no longer yields significant improvements in inertia.
* **Advantage**: Extremely intuitive and easy to compute.
* **Limitation**: Often the elbow point is subjective and not clearly defined in continuous datasets.
* **Code Snippet**:
  ```python
  k_values = range(2, 11)
  inertia_values = []

  for k in k_values:
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(X)
      inertia_values.append(kmeans.inertia_)

  # Plot the inertia values
  plt.figure(figsize=(6, 5))
  plt.plot(k_values, inertia_values, marker='o')
  plt.title('Elbow Method: Inertia vs. k')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Inertia')
  plt.show()
  ```

### 3. Silhouette Method (Score vs. $k$)
* **Description**: Evaluates the mean silhouette score for different values of $k$. The peak value represents the $k$ configuration that maximizes cluster cohesion and separation.
* **Advantage**: Directly quantifies cluster boundary separation and membership quality.
* **Limitation**: Computationally intensive for very large sample sizes.
* **Code Snippet**:
  ```python
  from sklearn.metrics import silhouette_score

  k_values = range(2, 11)
  silhouette_scores = []

  for k in k_values:
      kmeans = KMeans(n_clusters=k, random_state=42)
      y_kmeans = kmeans.fit_predict(X)
      silhouette_scores.append(silhouette_score(X, y_kmeans))

  # Plot the Silhouette Scores
  plt.figure(figsize=(6, 5))
  plt.plot(k_values, silhouette_scores, marker='o', color='green')
  plt.title('Silhouette Score vs. k')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Silhouette Score')
  plt.show()
  ```

### 4. Davies-Bouldin Index vs. $k$
* **Description**: Plots the Davies-Bouldin Index (DBI) for a range of $k$ values. The index represents the ratio of within-cluster distances to between-cluster distances. Lower values indicate better clustering.
* **Advantage**: Provides a strict numerical ratio representing compactness and separation.
* **Limitation**: Highly sensitive to cluster shapes and varying densities.
* **Code Snippet**:
  ```python
  from sklearn.metrics import davies_bouldin_score

  k_values = range(2, 11)
  davies_bouldin_indices = []

  for k in k_values:
      kmeans = KMeans(n_clusters=k, random_state=42)
      y_kmeans = kmeans.fit_predict(X)
      davies_bouldin_indices.append(davies_bouldin_score(X, y_kmeans))

  # Plot the Davies-Bouldin Index
  plt.figure(figsize=(6, 5))
  plt.plot(k_values, davies_bouldin_indices, marker='o', color='purple')
  plt.title('Davies-Bouldin Index vs. k')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Davies-Bouldin Index')
  plt.show()
  ```
