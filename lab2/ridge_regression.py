import numpy as np
from sklearn.metrics import accuracy_score

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None  # коэффициенты при признаках
        self.bias = None  # свободный член (сдвиг)

    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # Матрица регуляризации
        I = np.eye(X_bias.shape[1])
        I[0, 0] = 0

        A = X_bias.T @ X_bias + self.alpha * I
        b = X_bias.T @ y
        all_params = np.linalg.solve(A, b)

        # Разделяем bias и coefficients
        self.bias = all_params[0]
        self.coefficients = all_params[1:]

    def predict(self, X):
        return X @ self.coefficients + self.bias

    def predict_class(self, X):
        """Для классификации: преобразуем в ±1"""
        predictions = self.predict(X)
        return np.where(predictions >= 0, 1, -1)

    def score(self, X, y):
        y_pred = self.predict_class(X)
        return accuracy_score(y, y_pred)