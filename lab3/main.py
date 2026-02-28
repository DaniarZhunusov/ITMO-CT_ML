import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from feature_engineering import create_time_features
from ts_models import train_naive_model, train_holt_winters, train_sarima
from utils import evaluate_model, evaluate_forecast
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('boston_weather_data.csv')
df.columns = ['time', 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']

df['Date'] = pd.to_datetime(df['time'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# берем данные с 2018 по 2023
df = df['2018-03-01':'2023-03-01']

# Заполняем пропуски в tavg
df['tavg'].fillna(method='ffill', inplace=True)
df['tavg'].fillna(method='bfill', inplace=True)

# 1. Построение графика средней температуры
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df.index, df['tavg'], color='deepskyblue', linewidth=1.5)
ax1.set_title('Средняя температура в Бостоне 2018-2023', fontsize=14)
ax1.set_xlabel('Год', fontsize=12)
ax1.set_ylabel('Температура (°C)', fontsize=12)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Создаем временной ряд
ts_daily = df['tavg'].asfreq('D')

# 3. График автокорреляции (ACF) для проверки сезонности
fig2, ax2 = plt.subplots(figsize=(14, 5))
plot_acf(ts_daily, lags=365, alpha=0.05, ax=ax2)
ax2.set_title('Автокорреляция средней температуры (годовой цикл)', fontsize=14)
ax2.set_xlabel('Лаг (дни)', fontsize=12)
ax2.set_ylabel('Корреляция', fontsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. График частичной автокорреляции (PACF)
fig3, ax3 = plt.subplots(figsize=(14, 5))
plot_pacf(ts_daily, lags=100, alpha=0.05, ax=ax3, method='ywm')
ax3.set_title('Частичная автокорреляция (PACF)', fontsize=14)
ax3.set_xlabel('Лаг (дни)', fontsize=12)
ax3.set_ylabel('Корреляция', fontsize=12)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 1. Визуальный анализ
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 8))

# Исходный ряд
ax4a.plot(ts_daily.index, ts_daily, color='blue', linewidth=1, alpha=0.7)
ax4a.set_title('Средняя температура в Бостоне (исходный ряд)', fontsize=14)
ax4a.set_ylabel('Температура (°C)', fontsize=12)
ax4a.grid(True, alpha=0.3)

# Скользящее среднее для выделения тренда
ts_rolling = ts_daily.rolling(window=30).mean()  # 30-дневное скользящее среднее
ax4b.plot(ts_daily.index, ts_daily, color='blue', linewidth=1, alpha=0.3, label='Исходный')
ax4b.plot(ts_rolling.index, ts_rolling, color='red', linewidth=2, label='Среднее')
ax4b.set_title('Тренд (скользящее среднее)', fontsize=14)
ax4b.set_ylabel('Температура (°C)', fontsize=12)
ax4b.legend()
ax4b.grid(True, alpha=0.3)
ax4b.set_xlabel('Год', fontsize=12)

plt.tight_layout()
plt.show()

# 2. Годовой цикл
fig5, ax5 = plt.subplots(figsize=(14, 5))
# Группируем по месяцу и находим среднее
monthly_avg = ts_daily.groupby(ts_daily.index.month).mean()
ax5.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
ax5.set_title('Средняя температура по месяцам', fontsize=14)
ax5.set_xlabel('Месяц', fontsize=12)
ax5.set_ylabel('Температура (°C)', fontsize=12)
ax5.set_xticks(range(1, 13))
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Логарифмирование ряда ===
ts_log = np.log(ts_daily + 10)  # +10 чтобы избежать отрицательных значений при логарифмировании

fig6, ax6 = plt.subplots(figsize=(14, 5))
ax6.plot(ts_log.index, ts_log, color='green', linewidth=1.5)
ax6.set_title('Логарифмированная температура', fontsize=14)
ax6.set_xlabel('Год', fontsize=12)
ax6.set_ylabel('log', fontsize=12)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Дифференцирование ряда ===
ts_diff = ts_daily.diff().dropna()

fig7, ax7 = plt.subplots(figsize=(14, 5))
ax7.plot(ts_diff.index, ts_diff, color='orange', linewidth=1)
ax7.set_title('Дифференцирование температуры', fontsize=14)
ax7.set_xlabel('Год', fontsize=12)
ax7.set_ylabel('ΔТемпература (°C/день)', fontsize=12)
ax7.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Извлечение признаков
ml_df_clean = create_time_features(ts_daily)

# Сводная информация
print("\n" + "="*50)
print("Сводная информация")
print("="*50)
print(f"Анализируемый период: {df.index.min().date()} - {df.index.max().date()}")
print(f"Количество дней: {len(ts_daily)}")
print(f"Средняя температура: {ts_daily.mean():.2f}°C")
print(f"Минимальная температура: {ts_daily.min():.2f}°C")
print(f"Максимальная температура: {ts_daily.max():.2f}°C")
print(f"Стандартное отклонение: {ts_daily.std():.2f}°C")

# Анализ сезонности
print(f"\nСезонность:")
print(f"Самый теплый месяц: {monthly_avg.idxmax()} ({monthly_avg.max():.1f}°C)")
print(f"Самый холодный месяц: {monthly_avg.idxmin()} ({monthly_avg.min():.1f}°C)")

# Табличная регрессия
print("\n" + "=" * 50)
print("Решение задачи табличной регрессии")
print("=" * 50)

# Загружаем подготовленные данные с признаками
df_features = pd.read_csv('boston_weather_with_time_features.csv',
                          parse_dates=['Date'], index_col='Date')

print("Данные для табличной регрессии:")
print(f"Размер: {df_features.shape}")
print(f"Признаки: {len(df_features.columns)}")

print("\n1. Подготовка данных для регрессии")

# Определяем целевую переменную и признаки
df_features['target'] = df_features['tavg'].shift(-1)  # температура завтра

df_regression = df_features.dropna()

# Разделяем на признаки (X) и целевую переменную (y)
X = df_regression.drop(['tavg', 'target'], axis=1)  # все признаки кроме tavg и target
y = df_regression['target']  # целевая переменная - температура завтра

print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Датасет охватывает: {df_regression.index.min().date()} - {df_regression.index.max().date()}")

# Разделение на train/test (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nРазделение данных:")
print(f"Train: {len(X_train)} записей ({len(X_train) / len(X) * 100:.1f}%)")
print(f"Test:  {len(X_test)} записей ({len(X_test) / len(X) * 100:.1f}%)")
print(f"Train период: {X_train.index.min().date()} - {X_train.index.max().date()}")
print(f"Test период:  {X_test.index.min().date()} - {X_test.index.max().date()}")

print("\n2. Решение моделью Linear Regression")

from sklearn.linear_model import LinearRegression

# Создаем и обучаем модель
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Делаем прогнозы
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

print("\nОценка Linear Regression:")
train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred, "Train")
test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred, "Test")


# График результатов регрессии
fig, ax = plt.subplots(figsize=(14, 6))

# Прогнозы на тестовой выборке
ax.plot(y_test.index, y_test.values, label='Факт', linewidth=2, color='black')
ax.plot(y_test.index, y_test_pred, label=f'Linear Regression прогноз (MAE: {test_mae:.2f}°C)', alpha=0.8, color='blue')
ax.axvline(x=y_test.index[0], color='red', linestyle='-',
           linewidth=2, label='Начало теста')
ax.set_title('Прогнозы Linear Regression: Факт и Предсказание', fontsize=14)
ax.set_ylabel('Температура (°C)', fontsize=12)
ax.set_xlabel('Дата', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("Модели временных рядов")
print("="*50)

# Используем исходный ряд ts_daily
ts_values = ts_daily.values
train_size_ts = int(len(ts_values) * 0.8)

# Разделяем на обучающую и тестовую выборки для моделей временных рядов
train_ts = ts_values[:train_size_ts]
test_ts = ts_values[train_size_ts:]

# Даты для тестовой выборки (для визуализации)
test_dates = ts_daily.index[train_size_ts:train_size_ts + len(test_ts)]
split_date = test_dates[0]

print(f"Размер обучающей выборки: {len(train_ts)} дней")
print(f"Размер тестовой выборки:  {len(test_ts)} дней")

# Обучение моделей
naive_forecast = train_naive_model(train_ts, test_ts)
hw_forecast, hw_fit = train_holt_winters(train_ts, test_ts)
sarima_forecast, sarima_fit = train_sarima(train_ts, test_ts)


print("\n" + "="*50)
print("Оценка и визуализация результатов")
print("="*50)

print("\nМетрики на тестовой выборке:")
# Оцениваем все модели
metrics = {}
metrics['Наивная'] = evaluate_forecast('Наивная модель', test_ts, naive_forecast)
metrics['Хольта-Уинтерса'] = evaluate_forecast('Хольта-Уинтерса', test_ts, hw_forecast)
metrics['SARIMA'] = evaluate_forecast('SARIMA', test_ts, sarima_forecast)
metrics['Linear Regression'] = (test_mae, test_rmse)

# --- Сравнительный график прогнозов ---
fig, ax = plt.subplots(figsize=(16, 8))

# Фактические данные (последние 100 дней train + весь test)
plot_train_start = max(0, train_size_ts - 100)
ax.plot(ts_daily.index[plot_train_start:train_size_ts],
        ts_values[plot_train_start:train_size_ts],
        'gray', alpha=0.7, linewidth=1, label='Факт (train, последние 100 дн.)')
ax.plot(test_dates, test_ts, 'k-', linewidth=2.5, label='Факт (test)')

# Прогнозы моделей
ax.plot(test_dates, naive_forecast, 'orange', linestyle=':', linewidth=2, label='Наивная (среднее)')
ax.plot(test_dates, hw_forecast, 'blue', linestyle='--', linewidth=2, label='Хольта-Уинтерса')
ax.plot(test_dates, sarima_forecast, 'green', linestyle='--', linewidth=2, label='SARIMA')
ax.plot(y_test.index, y_test_pred, 'red', linestyle='-.', linewidth=1.5, alpha=0.8, label='Linear Regression')

# Граница train/test
ax.axvline(x=split_date, color='purple', linestyle='-', linewidth=2, alpha=0.7, label='Начало теста')

ax.set_title('Сравнение прогнозов моделей временных рядов', fontsize=16, pad=15)
ax.set_xlabel('Дата', fontsize=13)
ax.set_ylabel('Температура (°C)', fontsize=13)
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# --- График ошибок ---
fig2, ax = plt.subplots(1, 1, figsize=(14, 5))

ax.plot(test_dates, test_ts - naive_forecast, 'orange', linestyle=':', linewidth=1, label='Ошибка: Наивная')
ax.plot(test_dates, test_ts - hw_forecast, 'blue', linestyle='--', linewidth=1, label='Ошибка: Хольта-Уинтерса')
ax.plot(test_dates, test_ts - sarima_forecast, 'green', linestyle='--', linewidth=1, label='Ошибка: SARIMA')
lr_test_dates = y_test.index
lr_errors = y_test.values - y_test_pred
ax.plot(lr_test_dates, lr_errors, 'red', linestyle='-.', linewidth=1, alpha=0.8, label='Ошибка: Linear Regression')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(x=split_date, color='purple', linestyle='-', linewidth=2, alpha=0.3)
ax.set_title('Ошибки прогнозов по времени (Факт - Прогноз)', fontsize=14)
ax.set_ylabel('Ошибка (°C)', fontsize=12)
ax.set_xlabel('Дата', fontsize=12)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Находим модель с наименьшей MAE
best_model_name = min(metrics, key=lambda x: metrics[x][0])
best_mae = metrics[best_model_name][0]
print(f"Наилучшая модель по MAE: {best_model_name} ({best_mae:.2f}°C)")
