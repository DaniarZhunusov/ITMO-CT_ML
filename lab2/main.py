import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

from ridge_regression import RidgeRegression
from gradient_descent_classifier import GradientDescentClassifier

def load_data():
    data = pd.read_csv('data.csv')
    return data

def plot_smoothed_learning_curve(empirical_risk_history, window_size=20):
    risks = np.array(empirical_risk_history)

    if len(risks) >= window_size:
        smoothed_risks = np.convolve(risks, np.ones(window_size) / window_size, mode='valid')
        epochs_smoothed = np.arange(window_size - 1, len(risks))
    else:
        smoothed_risks = risks
        epochs_smoothed = np.arange(len(risks))

    plt.figure(figsize=(10, 6))
    plt.plot(empirical_risk_history, alpha=0.3, color='blue', label='Исходный эмпирический риск')
    plt.plot(epochs_smoothed, smoothed_risks, color='red', linewidth=2,
             label=f'Сглаженный эмпирический риск')
    plt.xlabel('Epoch')
    plt.ylabel('Эмпирический риск')
    plt.title('Кривая обучения: сглаженный эмпирический риск на тренировочном множестве')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return smoothed_risks

def cross_validation(X, y, n_splits=5, epochs=500, eval_test_every=20):
    """
    Кросс-валидация с построением кривой обучения на тестовом множестве
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Будем хранить историю точности для каждого разбиения
    all_test_accuracies = []
    ridge_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Обработка fold {fold + 1}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Обучаем Gradient Descent
        gd_model = GradientDescentClassifier(
            learning_rate=0.1,
            alpha=0.001,
            beta=0.001,
            loss_type='logistic',
            epochs=epochs,
            tol=1e-6
        )

        gd_model.fit(X_train, y_train, X_test, y_test,
                     eval_test_every=eval_test_every, logging=False)

        # Сохраняем историю точности на тесте
        fold_test_accuracies = [acc for _, acc in gd_model.test_accuracy_history]
        all_test_accuracies.append(fold_test_accuracies)

        # Обучаем Ridge регрессию и сохраняем результат
        ridge_model = RidgeRegression(alpha=0.1)
        ridge_model.fit(X_train, y_train)
        ridge_score = ridge_model.score(X_test, y_test)
        ridge_scores.append(ridge_score)

        print(f"Fold {fold + 1}: GD Final Test Acc = {fold_test_accuracies[-1]:.4f}, "
              f"Ridge Test Acc = {ridge_score:.4f}")

    return all_test_accuracies, ridge_scores

def plot_test_learning_curve_with_confidence(all_test_accuracies, ridge_score_mean,
                                             eval_test_every=20, epochs=500):
    """
    Построение кривой обучения на тестовом множестве с доверительным интервалом
    """
    # Преобразуем в numpy array
    test_accuracies = np.array(all_test_accuracies)

    # Вычисляем статистики
    mean_accuracies = np.mean(test_accuracies, axis=0)
    std_accuracies = np.std(test_accuracies, axis=0)

    # Доверительный интервал (95%)
    confidence_interval = 1.96 * std_accuracies / np.sqrt(len(all_test_accuracies))

    # Эпохи для которых у нас есть измерения
    measurement_epochs = np.arange(0, epochs + 1, eval_test_every)
    if len(measurement_epochs) > len(mean_accuracies):
        measurement_epochs = measurement_epochs[:len(mean_accuracies)]

    plt.figure(figsize=(12, 8))

    # Кривая обучения с доверительным интервалом
    plt.plot(measurement_epochs, mean_accuracies, 'b-', linewidth=2,
             label='Средняя точность на тесте (Gradient Descent)')
    plt.fill_between(measurement_epochs,
                     mean_accuracies - confidence_interval,
                     mean_accuracies + confidence_interval,
                     alpha=0.2, color='blue',
                     label='95% доверительный интервал')

    # Линия для Ridge регрессии
    plt.axhline(y=ridge_score_mean, color='red', linestyle='--',
                linewidth=2, label=f'Ridge Regression ({ridge_score_mean:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Точность на тестовом множестве')
    plt.title('Кривая обучения на тестовом множестве с доверительным интервалом\n'
              '(Перекрёстная проверка, 5 разбиений)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.show()

    return mean_accuracies, confidence_interval

def plot_regularization_paths_for_all_losses(X_train, y_train, n_points=50):
    """
    Построение графиков регуляризационных путей для всех трех функций потерь
    """
    print("\n" + "=" * 70)
    print("Построение графиков регуляризации для всех функций потерь...")

    # Функции потерь для анализа
    loss_types = ['hinge', 'logistic', 'square']
    loss_names = ['Hinge', 'Logistic', 'Square']

    # Создаем логарифмические пространства для коэффициентов регуляризации
    l1_values = np.logspace(-4, 2, n_points)
    l2_values = np.logspace(-4, 2, n_points)

    # Создаем большую фигуру для всех графиков
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    for i, (loss_type, loss_name) in enumerate(zip(loss_types, loss_names)):
        print(f"Анализ для функции потерь: {loss_name}")

        # Для L1 регуляризации (L2 = 0)
        l1_coefs = []
        for l1 in l1_values:
            model = GradientDescentClassifier(
                learning_rate=0.01,
                alpha=l1,
                beta=0.0,
                loss_type=loss_type,
                epochs=1000,
                tol=1e-6
            )
            model.fit(X_train, y_train, logging=False)
            l1_coefs.append(model.coefficients.copy())

        # Для L2 регуляризации (L1 = 0)
        l2_coefs = []
        for l2 in l2_values:
            model = GradientDescentClassifier(
                learning_rate=0.01,
                alpha=0.0,
                beta=l2,
                loss_type=loss_type,
                epochs=1000,
                tol=1e-6
            )
            model.fit(X_train, y_train, logging=False)
            l2_coefs.append(model.coefficients.copy())

        l1_coefs = np.array(l1_coefs)
        l2_coefs = np.array(l2_coefs)

        # График L1 регуляризации
        ax1 = axes[i, 0]
        plot_single_regularization_path(ax1, l1_values, l1_coefs, 'L1', loss_name)

        # График L2 регуляризации
        ax2 = axes[i, 1]
        plot_single_regularization_path(ax2, l2_values, l2_coefs, 'L2', loss_name)

    plt.tight_layout()
    plt.show()

    return l1_values, l2_values

def plot_single_regularization_path(ax, reg_values, coefs, reg_type, loss_name):
    """
    Построение одного графика регуляризационного пути
    """
    n_features = coefs.shape[1]

    # Выбираем топ-8 признаков для отображения
    n_to_plot = min(8, n_features)
    mean_abs_coefs = np.mean(np.abs(coefs), axis=0)
    top_features = np.argsort(mean_abs_coefs)[-n_to_plot:]

    colors = plt.cm.Set3(np.linspace(0, 1, n_to_plot))

    for j, feature_idx in enumerate(top_features):
        ax.plot(reg_values, coefs[:, feature_idx],
                color=colors[j], linewidth=2, alpha=0.8,
                label=f'Feature {feature_idx + 1}')

    ax.set_xscale('log')
    ax.set_xlabel(f'{reg_type} Regularization Coefficient', fontsize=10)
    ax.set_ylabel('Coefficient Value', fontsize=10)
    ax.set_title(f'{loss_name} Loss - {reg_type} Regularization', fontsize=12)
    ax.grid(True, alpha=0.3)

    if n_to_plot <= 6:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Анализ и аннотации для L1
    if reg_type == 'L1':
        non_zero_counts = np.sum(np.abs(coefs) > 1e-6, axis=1)
        if np.any(non_zero_counts < n_features):
            threshold_idx = np.where(non_zero_counts < n_features)[0]
            if len(threshold_idx) > 0:
                threshold_val = reg_values[threshold_idx[0]]
                ax.axvline(threshold_val, color='red', linestyle='--',
                           alpha=0.7, linewidth=1)

    # Анализ и аннотации для L2
    elif reg_type == 'L2':
        coef_norms = np.linalg.norm(coefs, axis=1)
        if len(coef_norms) > 1:
            shrinkage_idx = np.where(coef_norms < 0.8 * coef_norms[0])[0]
            if len(shrinkage_idx) > 0:
                threshold_val = reg_values[shrinkage_idx[0]]
                ax.axvline(threshold_val, color='red', linestyle='--',
                           alpha=0.7, linewidth=1)

def plot_combined_regularization_paths(X_train, y_train, n_points=50):
    """
    Альтернативный вариант: совмещенные графики для всех функций потерь
    """
    print("\n" + "=" * 70)
    print("Построение совмещенных графиков регуляризации...")

    loss_types = ['hinge', 'logistic', 'square']
    loss_names = ['Hinge', 'Logistic', 'Square']
    colors = ['blue', 'red', 'green']

    l1_values = np.logspace(-4, 2, n_points)
    l2_values = np.logspace(-4, 2, n_points)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Собираем данные о нормам коэффициентов
    for i, (loss_type, loss_name, color) in enumerate(zip(loss_types, loss_names, colors)):
        print(f"Обработка {loss_name} loss...")

        # L1 регуляризация
        l1_norms = []
        for l1 in l1_values:
            model = GradientDescentClassifier(
                learning_rate=0.01,
                alpha=l1,
                beta=0.0,
                loss_type=loss_type,
                epochs=1000,
                tol=1e-6
            )
            model.fit(X_train, y_train, logging=False)
            l1_norms.append(np.linalg.norm(model.coefficients))

        # L2 регуляризация
        l2_norms = []
        for l2 in l2_values:
            model = GradientDescentClassifier(
                learning_rate=0.01,
                alpha=0.0,
                beta=l2,
                loss_type=loss_type,
                epochs=1000,
                tol=1e-6
            )
            model.fit(X_train, y_train, logging=False)
            l2_norms.append(np.linalg.norm(model.coefficients))

        # Графики норм
        ax1.plot(l1_values, l1_norms, color=color, linewidth=2, label=loss_name)
        ax2.plot(l2_values, l2_norms, color=color, linewidth=2, label=loss_name)

    # Настройка графиков
    for ax, reg_type in zip([ax1, ax2], ['L1', 'L2']):
        ax.set_xscale('log')
        ax.set_xlabel(f'{reg_type} Regularization Coefficient')
        ax.set_ylabel('L2 Norm of Coefficients')
        ax.set_title(f'{reg_type} Regularization')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    # 1. Загрузка данных
    data = load_data()

    # 2. Создаем бинарную целевую переменную на основе Цены
    if 'Цена' in data.columns:
        # Используем медиану цены для разделения на два класса
        price_median = data['Цена'].median()
        data['target'] = np.where(data['Цена'] > price_median, 1, -1)
        print(f"Медианная цена: {price_median:.2f}")
        print(f"Дорогие товары (1): {(data['target'] == 1).sum()}")
        print(f"Дешевые товары (-1): {(data['target'] == -1).sum()}")
        print(f"Распределение классов: {data['target'].value_counts()}")

    # 3. Разделение на признаки и целевую переменную
    # Убираем цену и бренд из признаков
    X = data.drop(['target', 'Цена', 'Бренд'], axis=1)
    y = data['target']

    print(f"\nПризнаки: {X.shape[1]}, Объекты: {X.shape[0]}")
    print(f"Целевая переменная: {y.unique()}")

    # Преобразуем в numpy arrays для кросс-валидации
    X = X.values
    y = y.values

    # 4. Кросс-валидация
    print("\nЗапуск кросс-валидация")
    all_test_accuracies, ridge_scores = cross_validation(
        X, y, n_splits=5, epochs=500, eval_test_every=20
    )

    # 5. Построение кривой обучения на тестовом множестве
    ridge_score_mean = np.mean(ridge_scores)
    print(f"\nСредняя точность Ridge регрессии: {ridge_score_mean:.4f}")

    mean_accuracies, confidence_interval = plot_test_learning_curve_with_confidence(
        all_test_accuracies, ridge_score_mean, eval_test_every=20, epochs=500
    )

    # 6. Дополнительная информация
    final_gd_accuracy = np.mean([acc[-1] for acc in all_test_accuracies])
    print(f"\nИтоговые результаты:")
    print(f"Средняя финальная точность Gradient Descent: {final_gd_accuracy:.4f}")
    print(f"Средняя точность Ridge Regression: {ridge_score_mean:.4f}")
    print(f"Разница: {final_gd_accuracy - ridge_score_mean:.4f}")

    # 7. Обычное обучение для сравнения
    print("\n" + "=" * 50)
    print("Обучение на одном разбиении для сравнения:")

    # Разбиение на тренировочную (80%) и тестовую часть (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    gd_model = GradientDescentClassifier(
        learning_rate=0.1,
        alpha=0.001,
        beta=0.001,
        loss_type='logistic',
        epochs=500,
        tol=1e-6
    )

    gd_model.fit(X_train, y_train, X_test, y_test, eval_test_every=10, logging=True)

    # Построение кривой эмпирического риска
    plot_smoothed_learning_curve(gd_model.empirical_risk_history, window_size=15)

    # 8. Построение графиков регуляризационных путей
    plot_regularization_paths_for_all_losses(X_train, y_train, n_points=50)

    # 9. Дополнительные совмещенные графики
    plot_combined_regularization_paths(X_train, y_train, n_points=50)

if __name__ == "__main__":
    main()