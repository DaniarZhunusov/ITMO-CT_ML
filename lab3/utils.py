import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, dataset_name):
    """Функция для оценки модели регрессии"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name}:")
    print(f"  MAE:  {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²:   {r2:.4f}")
    print(f"  Средняя температура: {y_true.mean():.2f}°C")
    print(f"  Диапазон температур: {y_true.min():.1f}°C - {y_true.max():.1f}°C")

    return mae, rmse, r2

def evaluate_forecast(name, y_true, y_pred):
    """Вычисляет и печатает метрики для временных рядов."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"   {name}: MAE = {mae:.2f}°C, RMSE = {rmse:.2f}°C")
    return mae, rmse