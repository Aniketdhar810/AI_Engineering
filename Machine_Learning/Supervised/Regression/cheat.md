# Regression Models & Scikit-Learn Utilities Cheatsheet

This cheatsheet provides a quick comparison of standard regression types and commonly used functions in `scikit-learn` for preprocessing and evaluation.

---

## 1. Comparing Regression Types

| Model Name | Description & Modeling Equation | Code Syntax |
| :--- | :--- | :--- |
| **Simple Linear Regression** | **Purpose:** Predict a dependent variable based on one independent variable.<br><br>**Pros:** Easy to implement, interpret, and computationally efficient.<br>**Cons:** Prone to underfitting; cannot capture complex, non-linear relationships.<br><br>**Equation:** $y = b_0 + b_1x$ | ```python\nfrom sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression()\nmodel.fit(X, y)\n``` |
| **Polynomial Regression** | **Purpose:** Capture non-linear relationships by mapping features to polynomial space.<br><br>**Pros:** Fits non-linear data well.<br>**Cons:** Highly prone to overfitting at higher degrees.<br><br>**Equation:** $y = b_0 + b_1x + b_2x^2 + \dots + b_nx^n$ | ```python\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\n\npoly = PolynomialFeatures(degree=2)\nX_poly = poly.fit_transform(X)\nmodel = LinearRegression().fit(X_poly, y)\n``` |
| **Multiple Linear Regression** | **Purpose:** Predict a dependent variable based on multiple independent variables.<br><br>**Pros:** Accounts for multiple factors influencing the outcome.<br>**Cons:** Assumes a linear relationship between predictors and target.<br><br>**Equation:** $y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n$ | ```python\nfrom sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression()\nmodel.fit(X, y)\n``` |
| **Logistic Regression** | **Purpose:** Predict probabilities of categorical/binary outcomes.<br><br>**Pros:** Efficient and highly interpretable for classification.<br>**Cons:** Assumes a linear relationship between independent variables and log-odds.<br><br>**Equation:** $\log\left(\frac{p}{1-p}\right) = b_0 + b_1x_1 + \dots$ | ```python\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression()\nmodel.fit(X, y)\n``` |

---

## 2. Associated Functions & Preprocessing Utilities

### Data Splitting & Preprocessing

#### `train_test_split`
Splits the dataset into training and testing subsets to evaluate a model's generalizability.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### `StandardScaler`
Standardizes features by removing the mean and scaling to unit variance ($z = \frac{x - \mu}{\sigma}$).
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### Evaluation Metrics

#### `log_loss` (Logarithmic Loss / Cross-Entropy Loss)
Measures the performance of a classification model whose output is a probability value between 0 and 1.
```python
from sklearn.metrics import log_loss

loss = log_loss(y_true, y_pred_proba)
```

#### `mean_absolute_error` (MAE)
Computes the average of the absolute differences between the actual and predicted values.
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

#### `mean_squared_error` (MSE)
Computes the average of the squared differences between the actual and predicted values. Penalizes larger errors more heavily than MAE.
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

#### `root_mean_squared_error` (RMSE)
The square root of MSE. Keeps the error metric in the same units as the target variable.
```python
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```
> [!NOTE]
> In scikit-learn version `1.4+`, you can also import `root_mean_squared_error` directly:
> ```python
> from sklearn.metrics import root_mean_squared_error
> rmse = root_mean_squared_error(y_true, y_pred)
> ```

#### `r2_score` ($R^2$ Coefficient of Determination)
Indicates the proportion of variance in the dependent variable that is predictable from the independent variables (scale of $-\infty$ to $1$).
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```