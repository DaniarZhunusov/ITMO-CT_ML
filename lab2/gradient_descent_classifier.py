import numpy as np
from sklearn.metrics import accuracy_score

def loss_function(y_true, y_pred, loss_type='logistic'):
    margin = y_true * y_pred  # Отступ
    if loss_type == 'logistic':
        return np.log(1 + np.exp(-margin))
    elif loss_type == 'hinge':
        return np.maximum(0, 1 - margin)
    elif loss_type == 'square':
        return (1 - margin) ** 2
    else:
        raise ValueError(f'Invalid loss function: {loss_type}')

class GradientDescentClassifier:
    def __init__(self, learning_rate=0.1, alpha=0.01, beta=0.01,
                 loss_type='logistic', epochs=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self.epochs = epochs
        self.tol = tol

        self.coefficients = None
        self.bias = None
        self.loss_history = []
        self.empirical_risk_history = []
        self.test_accuracy_history = []

    def _loss_gradient(self, y_true, y_pred):
        margin = y_true * y_pred

        if self.loss_type == 'logistic':
            exp_term = np.exp(-margin)
            return -y_true * exp_term / (1 + exp_term)
        elif self.loss_type == 'hinge':
            return np.where(margin < 1, -y_true, 0)
        elif self.loss_type == 'square':
            return -2 * y_true * (1 - margin)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _regularization(self):
        return self.alpha * np.sum(np.abs(self.coefficients)) + self.beta * np.sum(self.coefficients ** 2) / 2

    def _regularization_gradient(self):
        return self.alpha * np.sign(self.coefficients) + self.beta * self.coefficients

    def fit(self, X, y, X_test=None, y_test=None, eval_test_every=10, logging=False):
        n_samples, n_features = X.shape

        self.coefficients = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        self.loss_history = []
        self.empirical_risk_history = []
        self.test_accuracy_history = []

        for epoch in range(self.epochs):
            # Прямое распространение
            y_pred = self._predict_raw(X)

            # Вычисление потерь
            empirical_risk = np.mean(loss_function(y, y_pred, self.loss_type))
            reg_penalty = self._regularization()
            total_loss = empirical_risk + reg_penalty

            # Сохраняем историю ошибок
            self.loss_history.append(total_loss)
            self.empirical_risk_history.append(empirical_risk)

            # Сохраняем точность на тестовой выборке
            if X_test is not None and y_test is not None and epoch % eval_test_every == 0:
                test_accuracy = self.score(X_test, y_test)
                self.test_accuracy_history.append((epoch, test_accuracy))

            # Проверка критерия остановки
            if len(self.loss_history) > 1:
                loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
                if loss_change < self.tol:
                    if logging:
                        print(f"Ранняя остановка на epoch {epoch}")
                    break

            # Обратное распространение
            loss_grad = self._loss_gradient(y, y_pred)
            grad_coefficients = (X.T @ loss_grad) / n_samples + self._regularization_gradient()
            grad_bias = np.mean(loss_grad)

            # Градиентный спуск
            self.coefficients -= self.learning_rate * grad_coefficients
            self.bias -= self.learning_rate * grad_bias

            # Логирование
            if logging and epoch % 100 == 0:
                train_accuracy = self.score(X, y)
                test_accuracy = self.score(X_test, y_test) if X_test is not None else 0
                print(
                    f"Epoch {epoch}, Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

        # Добавляем финальную точность на тесте
        if X_test is not None and y_test is not None:
            final_test_accuracy = self.score(X_test, y_test)
            self.test_accuracy_history.append((self.epochs, final_test_accuracy))

        if logging:
            final_accuracy = self.score(X, y)
            print(f"Обучение завершено за {len(self.loss_history)} epoch")
            print(f"Финальная точность на тренировочных данных: {final_accuracy:.4f}")

    def _predict_raw(self, X):
        return X @ self.coefficients + self.bias

    def predict(self, X):
        raw_predictions = self._predict_raw(X)
        return np.where(raw_predictions >= 0, 1, -1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)