import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_naive_model(train_ts, test_ts):
    """Наивная модель с преобразованием 'логарифмирование + дифференцирование'"""
    print("\nНаивная модель (с преобразованием log+diff)")

    # Параметр сдвига для логарифмирования
    log_shift = 10

    # Логарифмирование обучающих данных
    train_log = np.log(train_ts + log_shift)

    # Дифференцирование логарифмированных данных
    train_log_diff = np.diff(train_log)

    # Обучение на преобразованных данных (вычисление среднего)
    train_log_diff_clean = train_log_diff[np.isfinite(train_log_diff)]

    if len(train_log_diff_clean) > 0:
        mean_log_diff = train_log_diff_clean.mean()
        print(f"   Среднее преобразованного ряда: {mean_log_diff:.6f}")

        # Прогноз
        forecast_log_diff = np.full(len(test_ts), mean_log_diff)

        # Обратное преобразование прогнозов
        last_log_value = train_log[-1]

        # Обратное дифференцирование: y_t = y_{t-1} + diff_t
        integrated = np.zeros(len(test_ts))
        integrated[0] = last_log_value + forecast_log_diff[0]
        for i in range(1, len(integrated)):
            integrated[i] = integrated[i - 1] + forecast_log_diff[i]

        # Обратное логарифмирование
        naive_forecast_transformed = np.exp(integrated) - log_shift

        print(f"   Прогнозное значение: {naive_forecast_transformed[0]:.2f}°C")
        print(f"   Диапазон прогнозов: {naive_forecast_transformed.min():.1f} - {naive_forecast_transformed.max():.1f}°C")

        return naive_forecast_transformed


def train_holt_winters(train_ts, test_ts, transformation='diff'):
    """Хольта-Уинтерс с преобразованиями"""
    print(f"\nХольта-Уинтерс (с преобразованием '{transformation}')")

    train_diff = np.diff(train_ts)

    if len(train_diff) > 0:
        # Обучаем на дифференцированных данных
        hw_model = ExponentialSmoothing(
            train_diff,
            trend='add',
            seasonal='add',
            seasonal_periods=365
        )
        hw_fit = hw_model.fit()
        hw_forecast_transformed = hw_fit.forecast(steps=len(test_ts))

        # Обратное преобразование прогнозов (обратное дифференцирование)
        last_value = train_ts[-1]
        hw_forecast = np.cumsum(hw_forecast_transformed) + last_value

        print(f"   Прогноз после обратного преобразования: {hw_forecast[0]:.2f}°C")

        return hw_forecast, hw_fit


def train_sarima(train_ts, test_ts):
    """SARIMA"""
    print("\nSARIMA")

    # Дифференцирование
    train_diff = np.diff(train_ts)

    if len(train_diff) > 0:
        try:
            sarima_model = SARIMAX(
                train_diff,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_fit = sarima_model.fit(disp=False)
            forecast_diff = sarima_fit.forecast(steps=len(test_ts))

            # Обратное преобразование
            last_value = train_ts[-1]
            sarima_forecast = np.cumsum(forecast_diff) + last_value

            print(f"   Прогноз: {sarima_forecast[0]:.2f}°C")

            return sarima_forecast, sarima_fit
        except Exception as e:
            print(f"   Ошибка: {e}")

    return np.full_like(test_ts, train_ts.mean())
