# Unsupervised Learning Reference Guide

This reference guide provides a summary of key unsupervised learning models (for dimensionality reduction and clustering) and associated utility functions commonly used in machine learning workflows.

---

## Unsupervised Learning Models

### Dimensionality Reduction

#### 1. UMAP (Uniform Manifold Approximation and Projection)
* **Description**: A non-linear dimensionality reduction technique based on manifold learning and topological data analysis.
* **Pros**: 
  * Excellent runtime performance on large datasets.
  * Preserves both local and global data structures better than many alternatives (like t-SNE).
* **Cons**: 
  * Sensitive to hyperparameters (especially `n_neighbors` and `min_dist`).
  * Non-deterministic without a fixed random state.
* **Applications**: Data visualization, exploratory data analysis, and feature extraction.
* **Key Hyperparameters**:
  * `n_neighbors` (default = `15`): Controls the size of the local neighborhood. Larger values capture more global structure, while smaller values preserve local detail.
  * `min_dist` (default = `0.1`): Controls how closely points are packed together in the low-dimensional space.
  * `n_components` (default = `2`): The dimensionality of the target embedding space.
* **Code Syntax**:
  ```python
  from umap.umap_ import UMAP
  
  umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
  ```

---

#### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
* **Description**: A non-linear dimensionality reduction technique designed to represent high-dimensional data in a low-dimensional space (typically 2D or 3D) by preserving local pairwise similarities.
* **Pros**: 
  * Outstanding for visualizing clusters and high-dimensional manifolds.
* **Cons**: 
  * Computationally expensive and memory-intensive for large datasets.
  * Preserves local structure but does not guarantee preservation of global structures.
  * Prone to overfitting and sensitive to perplexity.
* **Applications**: High-dimensional data visualization, anomaly detection support.
* **Key Hyperparameters**:
  * `n_components` (default = `2`): Number of dimensions for the output.
  * `perplexity` (default = `30`): Balances attention between local and global aspects of the data. Usually set between 5 and 50.
  * `learning_rate` (default = `200`): Controls the step size during gradient descent optimization.
* **Code Syntax**:
  ```python
  from sklearn.manifold import TSNE
  
  tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
  ```

---

#### 3. PCA (Principal Component Analysis)
* **Description**: A linear dimensionality reduction technique that identifies orthogonal directions (principal components) along which the variance of the data is maximized.
* **Pros**: 
  * Easy to interpret and mathematically straightforward.
  * Highly efficient and reduces noise in data.
* **Cons**: 
  * Linear assumption; cannot capture complex non-linear relationships in data.
  * May lose critical information if the underlying structure is non-linear.
* **Applications**: Feature extraction, data compression, noise reduction, and preprocessing.
* **Key Hyperparameters**:
  * `n_components` (default = `2`): Number of principal components to retain.
  * `whiten` (default = `False`): Scales the components to ensure uncorrelated outputs with unit variance.
  * `svd_solver` (default = `'auto'`): The algorithm used to compute the singular value decomposition.
* **Code Syntax**:
  ```python
  from sklearn.decomposition import PCA
  
  pca = PCA(n_components=2)
  ```

---

### Clustering Algorithms

#### 4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
* **Description**: A density-based clustering algorithm that groups together points that are close to each other while marking points in low-density regions as outliers.
* **Pros**: 
  * Automatically identifies noise/outliers.
  * Does not require the number of clusters to be specified in advance.
  * Can find clusters of arbitrary shapes.
* **Cons**: 
  * Struggles to cluster datasets with varying densities.
  * Highly sensitive to the distance metric and parameters (`eps`, `min_samples`).
* **Applications**: Anomaly detection, spatial data clustering, and geographic analysis.
* **Key Hyperparameters**:
  * `eps` (default = `0.5`): The maximum distance between two points to be considered neighbors.
  * `min_samples` (default = `5`): The minimum number of samples in a neighborhood to form a core point/dense region.
* **Code Syntax**:
  ```python
  from sklearn.cluster import DBSCAN
  
  dbscan = DBSCAN(eps=0.5, min_samples=5)
  ```

---

#### 5. HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
* **Description**: An extension of DBSCAN that converts it into a hierarchical clustering algorithm and extracts flat clusters based on cluster stability.
* **Pros**: 
  * Successfully handles clusters of varying densities.
  * Simpler parameter selection than standard DBSCAN.
* **Cons**: 
  * Can be slower and more computationally intensive than DBSCAN.
* **Applications**: Large datasets, complex clustering problems, and automated cluster discovery.
* **Key Hyperparameters**:
  * `min_cluster_size` (default = `5`): The minimum size a group of points must be to be considered a cluster.
  * `min_samples` (default = `10`): The number of samples in a neighborhood for a point to be considered a core point (defaults to `min_cluster_size` if not specified).
* **Code Syntax**:
  ```python
  import hdbscan
  
  clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
  ```

---

#### 6. K-Means Clustering
* **Description**: A centroid-based partitioning algorithm that groups data into $k$ distinct, non-overlapping clusters by minimizing the distance between data points and their corresponding cluster centroid.
* **Pros**: 
  * Simple to understand and implement.
  * Highly scalable and computationally efficient.
* **Cons**: 
  * Requires specifying the number of clusters ($k$) in advance.
  * Highly sensitive to initial centroid placement and outliers.
  * Assumes spherical and similarly sized clusters.
* **Applications**: Customer segmentation, image compression, and pattern recognition.
* **Key Hyperparameters**:
  * `n_clusters` (default = `8`): Number of clusters to form.
  * `init` (default = `'k-means++'`): Initialization method. `'k-means++'` selects initial cluster centers in a smart way to speed up convergence.
  * `n_init` (default = `10`): Number of times the algorithm will run with different centroid seeds.
* **Code Syntax**:
  ```python
  from sklearn.cluster import KMeans
  
  kmeans = KMeans(n_clusters=3)
  ```

---

## Associated Utility & Helper Functions

Below are the commonly used helper functions for data generation, visualization, spatial manipulation, and model diagnostics.

### Data Generation

#### `make_blobs`
Generates isotropic Gaussian blobs for testing clustering algorithms.
```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=42)
```

#### `multivariate_normal`
Generates random samples from a multivariate normal distribution.
```python
from numpy.random import multivariate_normal

samples = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)
```

---

### Data Visualization & Mapping

#### `plotly.express.scatter_3d`
Creates interactive 3D scatter plots.
```python
import plotly.express as px

fig = px.scatter_3d(df, x='x', y='y', z='z')
fig.show()
```

#### `geopandas.GeoDataFrame`
Creates a GeoDataFrame from a standard Pandas DataFrame for spatial operations.
```python
import geopandas as gpd

gdf = gpd.GeoDataFrame(df, geometry='geometry')
```

#### `geopandas.to_crs`
Transforms the Coordinate Reference System (CRS) of a GeoDataFrame.
```python
gdf = gdf.to_crs(epsg=3857)
```

#### `contextily.add_basemap`
Adds a background map (basemap) to a GeoDataFrame plot for contextual visualization.
```python
import contextily as ctx

ax = gdf.plot(figsize=(10, 10))
ctx.add_basemap(ax)
```

---

### Model Diagnostics

#### `pca.explained_variance_ratio_`
An attribute of PCA that returns the proportion of the dataset's variance explained by each principal component.
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
variance_ratio = pca.explained_variance_ratio_
```
