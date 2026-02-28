import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# 1. Реализация методов отбора признаков

# 1.1 Встроенный метод (Lasso-регуляризация)
class LassoEmbeddedSelector(BaseEstimator, TransformerMixin):
    """Реализация Lasso для отбора признаков."""

    def __init__(self, alpha=0.01, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.selected_ = None

    def _soft_threshold(self, z):
        """Векторизованная функция мягкого порога."""
        return np.sign(z) * np.maximum(np.abs(z) - self.alpha, 0)

    def fit(self, X, y):
        """Обучение Lasso модели с координатным спуском."""
        # Преобразуем в матрицу
        X_dense = X.toarray() if hasattr(X, 'toarray') else X

        # Стандартизация
        # Центрируем данные, вычисляем среднее
        X_mean, y_mean = X_dense.mean(0), y.mean()
        X_centered = X_dense - X_mean
        y_centered = y - y_mean

        # Масштабирование
        X_std = X_centered.std(0)
        X_std[X_std == 0] = 1
        X_norm = X_centered / X_std

        # Координатный спуск
        n, p = X_norm.shape
        beta = np.zeros(p)

        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j = y_centered - X_norm @ beta + X_norm[:, j] * beta[j]
                beta[j] = self._soft_threshold(X_norm[:, j] @ r_j / n)

            # Проверка сходимости
            if np.abs(beta - beta_old).max() < self.tol:
                break

        # Обратное масштабирование и сохранение
        self.coef_ = beta / X_std
        self.selected_ = np.where(np.abs(self.coef_) > 1e-8)[0]
        self.intercept_ = y_mean - X_mean @ self.coef_

        return self

    def transform(self, X):
        """Отбор признаков."""
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        return X_dense[:, self.selected_]

    def get_features(self, feature_names, top_k=30):
        """Возвращает топ-K признаков с коэффициентами."""
        idx = self.selected_[np.argsort(np.abs(self.coef_[self.selected_]))[::-1][:top_k]]
        return {
            'indices': idx,
            'features': feature_names[idx],
            'coefficients': self.coef_[idx]
        }


# 1.2 Обёрточный метод (рекурсивное исключение)

class RFEWrappedSelector:
    """Реализация RFE."""

    def __init__(self, estimator, n_features_to_select=30, step=10):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.support_ = None
        self.ranking_ = None

    def fit(self, X, y):
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        n_features = X_dense.shape[1]

        # Инициализация
        self.support_ = np.ones(n_features, dtype=bool)
        self.ranking_ = np.ones(n_features, dtype=int)

        # Рекурсивное исключение
        while np.sum(self.support_) > self.n_features_to_select:
            # Обучаем модель
            self.estimator.fit(X_dense[:, self.support_], y)

            if hasattr(self.estimator, 'coef_'):
                # Для линейных моделей
                coef = self.estimator.coef_.ravel()
                importance = np.abs(coef)
            else:
                # Для tree-based моделей
                importance = self.estimator.feature_importances_

            # Находим наименее важные признаки
            indices = np.where(self.support_)[0]
            least_important = np.argsort(importance)[:self.step]

            # Исключаем признаки
            for idx in least_important:
                self.support_[indices[idx]] = False
                self.ranking_[indices[idx]] = n_features - np.sum(~self.support_)

        return self

    def transform(self, X):
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        return X_dense[:, self.support_]


# 1.3 Фильтрующий метод (хи-квадрат)
class ChiSquareFilterSelector(BaseEstimator, TransformerMixin):
    """Реализация фильтрующего метода на основе хи-квадрат."""

    def __init__(self, k=30):
        self.k = k
        self.scores_ = None
        self.selected_ = None

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        y = np.asarray(y)
        chi2_scores = np.zeros(n_features)

        iterator = range(n_features)
        if verbose:
            print(f"Хи-квадрат: обработка {n_features} признаков...")
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator)
            except:
                pass

        for i in iterator:
            # Оригинальная логика расчета
            col = X[:, i].toarray().flatten()

            # Таблица сопряженности
            table = np.array([
                [(y[col == 0] == 0).sum(), (y[col == 0] == 1).sum()],
                [(y[col > 0] == 0).sum(), (y[col > 0] == 1).sum()]
            ])

            total = table.sum()
            if total == 0:
                continue

            # Ожидаемые значения
            expected = (table.sum(axis=1).reshape(-1, 1) * table.sum(axis=0) / total)
            expected[expected == 0] = 1e-10

            # Хи-квадрат
            chi2_scores[i] = ((table - expected) ** 2 / expected).sum()

        self.scores_ = chi2_scores
        self.selected_ = np.argsort(chi2_scores)[-min(self.k, n_features):][::-1]

        if verbose:
            print(f"Выбрано {len(self.selected_)} признаков")

        return self

    def transform(self, X):
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        return X_dense[:, self.selected_]

    def get_support(self, indices=True):
        return self.selected_ if indices else (self.scores_ != 0)