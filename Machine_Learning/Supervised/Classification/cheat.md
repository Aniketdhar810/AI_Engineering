
## 1. Common Supervised Learning Models

| Model Name | Description & Key Hyperparameters | Code Syntax |
| :--- | :--- | :--- |
| **One vs One Classifier** | **Process:** Trains one classifier for each pair of classes.<br><br>**Key Hyperparameters:**<br>- `estimator`: Base classifier (e.g., `LogisticRegression()`) <br><br>**Pros:** Effective for small datasets.<br>**Cons:** Computationally expensive for large numbers of classes.<br>**Applications:** Multiclass classification with a small number of classes. | ```python\nfrom sklearn.multiclass import OneVsOneClassifier\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = OneVsOneClassifier(LogisticRegression())\n``` |
| **One vs All (Rest) Classifier** | **Process:** Trains one classifier per class, distinguishing that class from all others.<br><br>**Key Hyperparameters:**<br>- `estimator`: Base classifier (e.g., `LogisticRegression()`) <br><br>**Pros:** Simple, scalable, and highly parallelizable.<br>**Cons:** Can struggle with highly imbalanced datasets.<br>**Applications:** General multiclass classification (e.g., image recognition). | ```python\nfrom sklearn.multiclass import OneVsRestClassifier\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = OneVsRestClassifier(LogisticRegression())\n``` |
| **Decision Tree Classifier** | **Process:** Tree-based model that recursively splits data based on feature thresholds.<br><br>**Key Hyperparameters:**<br>- `max_depth`: Maximum depth of the tree (prevents overfitting)<br>- `criterion`: Split quality measure (`gini`, `entropy`, `log_loss`) <br><br>**Pros:** Highly interpretable, visualizable, and requires no feature scaling.<br>**Cons:** Highly prone to overfitting if not pruned or limited in depth.<br>**Applications:** Risk assessment, tabular classification. | ```python\nfrom sklearn.tree import DecisionTreeClassifier\n\nmodel = DecisionTreeClassifier(max_depth=5)\n``` |
| **Decision Tree Regressor** | **Process:** Tree-based model used for predicting continuous targets by averaging target values in leaf nodes.<br><br>**Key Hyperparameters:**<br>- `max_depth`: Maximum depth of the tree <br><br>**Pros:** Captures non-linear relationships, easy to interpret.<br>**Cons:** Sensitive to noise; can overfit easily.<br>**Applications:** General continuous prediction (e.g., real estate valuation). | ```python\nfrom sklearn.tree import DecisionTreeRegressor\n\nmodel = DecisionTreeRegressor(max_depth=5)\n``` |
| **Linear SVM Classifier** | **Process:** Finds the optimal separating hyperplane that maximizes the margin between classes.<br><br>**Key Hyperparameters:**<br>- `C`: Regularization strength (smaller values lead to a wider margin but more training violations)<br>- `kernel`: Function type (`linear`, `poly`, `rbf`, `sigmoid`) <br><br>**Pros:** Highly effective in high-dimensional spaces.<br>**Cons:** Does not scale well to very large datasets; sensitive to noise.<br>**Applications:** Text classification, spam detection, image recognition. | ```python\nfrom sklearn.svm import SVC\n\nmodel = SVC(kernel='linear', C=1.0)\n``` |
| **K-Nearest Neighbors Classifier** | **Process:** Classifies a data point based on the majority vote of its $k$ nearest neighbors.<br><br>**Key Hyperparameters:**<br>- `n_neighbors`: Number of neighbors ($k$)<br>- `weights`: Weighting of neighbors (`uniform`, `distance`) <br><br>**Pros:** Simple to understand, no training phase (lazy learner).<br>**Cons:** Slow during inference on large datasets; sensitive to irrelevant features.<br>**Applications:** Recommendation engines, basic pattern recognition. | ```python\nfrom sklearn.neighbors import KNeighborsClassifier\n\nmodel = KNeighborsClassifier(n_neighbors=5, weights='uniform')\n``` |
| **Random Forest Regressor** | **Process:** An ensemble of many independent decision trees (bagging) to reduce variance.<br><br>**Key Hyperparameters:**<br>- `n_estimators`: Number of trees in the forest<br>- `max_depth`: Maximum depth of each tree <br><br>**Pros:** Robust to overfitting, handles high-dimensional data well.<br>**Cons:** Harder to interpret than single trees; slower to train and predict.<br>**Applications:** Sales forecasting, stock price prediction. | ```python\nfrom sklearn.ensemble import RandomForestRegressor\n\nmodel = RandomForestRegressor(n_estimators=100, max_depth=5)\n``` |
| **XGBoost Regressor** | **Process:** Gradient boosted decision trees built sequentially to correct errors of previous trees.<br><br>**Key Hyperparameters:**<br>- `n_estimators`: Number of boosting rounds<br>- `learning_rate`: Step size shrinkage (`eta`) to prevent overfitting<br>- `max_depth`: Maximum depth of trees <br><br>**Pros:** Extremely high performance, speed, and accuracy.<br>**Cons:** Complex to tune; computationally expensive.<br>**Applications:** Tabular predictive modeling, competitive machine learning. | ```python\nimport xgboost as xgb\n\nmodel = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)\n``` |

---

## 2. Associated Utility Functions

### Data Preprocessing & Encoding

#### `OneHotEncoder`
Converts categorical features into a one-hot numeric matrix.
```python
from sklearn.preprocessing import OneHotEncoder

# Note: 'sparse_output' replaces the deprecated 'sparse' parameter in newer sklearn versions
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(categorical_data)
```

#### `LabelEncoder`
Encodes target labels with value between $0$ and $n\_classes-1$ (used only for targets, not features).
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
```

#### `normalize`
Scales individual samples to have unit norm ($L2$ or $L1$).
```python
from sklearn.preprocessing import normalize

normalized_data = normalize(data, norm='l2')
```

---

### Model Evaluation & Visualization

#### `accuracy_score`
Computes the fraction of correctly predicted classifications.
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
```

#### `roc_auc_score`
Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_score)
```

#### `plot_tree`
Generates a visual diagram of a trained decision tree.
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(model, max_depth=3, filled=True)
plt.show()
```

---

### Sample Weighting

#### `compute_sample_weight`
Estimates sample weights for class imbalance correction.
```python
from sklearn.utils.class_weight import compute_sample_weight

weights = compute_sample_weight(class_weight='balanced', y=y)
```
