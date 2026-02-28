import pandas as pd
import numpy as np


def create_time_features(ts_daily):
    print("Извлечение признаков")

    # Создаем DataFrame для ML
    ml_df = pd.DataFrame({'tavg': ts_daily})

    # 1. Основные признаки
    ml_df['day_of_year'] = ml_df.index.dayofyear  # День года
    ml_df['day_of_week'] = ml_df.index.dayofweek  # День недели
    ml_df['day_of_month'] = ml_df.index.day  # День месяца
    ml_df['week_of_year'] = ml_df.index.isocalendar().week  # Неделя года
    ml_df['month'] = ml_df.index.month  # Месяц
    ml_df['quarter'] = ml_df.index.quarter  # Квартал
    ml_df['year'] = ml_df.index.year  # Год

    # 2. Дополнительные признаки
    ml_df['is_weekend'] = (ml_df['day_of_week'] >= 5).astype(int)  # Выходные
    ml_df['is_month_start'] = ml_df.index.is_month_start.astype(int)  # Начало месяца
    ml_df['is_month_end'] = ml_df.index.is_month_end.astype(int)  # Конец месяца

    # 3. Сезонные признаки
    # Синус и косинус для циклических признаков
    ml_df['month_sin'] = np.sin(2 * np.pi * ml_df['month'] / 12)
    ml_df['month_cos'] = np.cos(2 * np.pi * ml_df['month'] / 12)
    ml_df['day_of_year_sin'] = np.sin(2 * np.pi * ml_df['day_of_year'] / 365.25)
    ml_df['day_of_year_cos'] = np.cos(2 * np.pi * ml_df['day_of_year'] / 365.25)

    # 4. Лагированные значения
    for lag in [1, 2, 3, 7, 14]:  # лаги на 1, 2, 3, 7, 14 дней
        ml_df[f'tavg_lag_{lag}'] = ml_df['tavg'].shift(lag)

    # 5. Скользящие статистики
    ml_df['tavg_rolling_mean_7'] = ml_df['tavg'].rolling(window=7).mean().shift(1)
    ml_df['tavg_rolling_std_7'] = ml_df['tavg'].rolling(window=7).std().shift(1)

    ml_df_clean = ml_df.dropna()

    # Вывод информации
    print(f"\nИзвлечено признаков: {len(ml_df_clean.columns)}")
    print(f"Наблюдений: {len(ml_df_clean)}")
    print(f"Период: {ml_df_clean.index.min().date()} - {ml_df_clean.index.max().date()}")

    # Сохранение
    ml_df_clean.to_csv('boston_weather_with_time_features.csv')
    print(f"\nДанные сохранены в 'boston_weather_with_time_features.csv'")

    return ml_df_clean