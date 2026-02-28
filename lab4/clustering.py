import numpy as np

# 4. Кластеризация (K-Means)

class KMeansClustering:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        """Обучение K-Means."""
        n_samples = X.shape[0]

        # Устанавливаем random seed если задан
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Инициализация центроидов (случайные точки)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        # Основной цикл
        for iteration in range(self.max_iter):
            # 2. Вычисление расстояний и назначение кластеров
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # 3. Вычисление инерции (сумма квадратов расстояний)
            self.inertia_ = np.sum(np.min(distances ** 2, axis=1))

            # 4. Пересчет центроидов с проверкой на пустые кластеры
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    # Пустой кластер - реинициализируем случайной точкой
                    new_centroids.append(X[np.random.randint(n_samples)])
            new_centroids = np.array(new_centroids)

            # 5. Проверка сходимости
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)

            # Обновляем центроиды
            self.centroids = new_centroids

            # Если центроиды почти не изменились - останавливаемся
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        return self

    def predict(self, X):
        """Предсказание кластеров для новых данных."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Обучение и предсказание за один вызов."""
        self.fit(X)
        return self.labels
