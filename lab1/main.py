import random
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
import json

PRODUCT_FEATURES = {
    'Бренд': 'STRING',
    'Тип': 'STRING',
    'Цвет': 'STRING',
    'Пол': 'STRING',
    'Сезон': 'STRING',
    'Цена': 'NUMERIC',
    'Старая цена': 'NUMERIC',
    'Материал верха основной': 'STRING',
    'Материал подкладки основной': 'STRING',
    'Материал стельки': 'STRING',
    'Материал подошвы': 'STRING',
    'Перфорация': 'STRING',
    'Тип застежки': 'STRING',
    'Высота каблука (мм)': 'NUMERIC',
    'Высота верха обуви (мм)': 'NUMERIC',
    'Полнота': 'NUMERIC',
}

TRANSLATION = {
    'Бренд': 'Brand',
    'Тип': 'Type',
    'Цвет': 'Color',
    'Пол': 'Gender',
    'Сезон': 'Season',
    'Цена': 'Price',
    'Старая цена': 'Old price',
    'Материал верха основной': 'Upper Material',
    'Материал подкладки основной': 'Lining Material',
    'Материал стельки': 'Insole Material',
    'Материал подошвы': 'Sole Material',
    'Перфорация': 'Perforation',
    'Тип застежки': 'Closure Type',
    'Высота каблука (мм)': 'Heel Height (mm)',
    'Высота верха обуви (мм)': 'Upper Height (mm)',
    'Полнота': 'Width',
}

category_values = {
    key: set() for key in [
        'Бренд', 'Тип', 'Цвет', 'Пол', 'Сезон',
        'Материал верха основной', 'Материал подкладки основной',
        'Материал стельки', 'Материал подошвы', 'Перфорация',
        'Тип застежки'
    ]
}


# Получение HTML
def fetch_webpage(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve page, status code: {response.status_code}")
    except Exception as e:
        print(f"Error getting {url}: {e}")
    return None


# Парсинг всех страниц
def all_products(base_url, num_pages):
    all_data = []
    for page in range(1, num_pages + 1):
        print(f"Парсим страницу {page}")
        url = f"{base_url}?page={page}"
        html = fetch_webpage(url)
        if html:
            boots_data = catalog_page(html)
            all_data.extend(boots_data)
        else:
            print(f"Failed to retrieve page {page}")
    return all_data


# Парсинг страницы каталога
def catalog_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    boots = soup.find_all('div', class_='v-catalog-card relative')
    data = []

    for idx, sneaker in enumerate(boots, start=1):
        product_link_tag = sneaker.find('a', href=True)
        if not product_link_tag:
            continue

        product_link = product_link_tag['href']
        product_url = f"https://thomas-muenz.ru{product_link}"
        print(f"[{idx}/{len(boots)}] Обработка: {product_url}")

        time.sleep(random.uniform(1.5, 3.0))
        product_details = product_info(product_url)
        data.append({i: product_details.get(i, 'Unknown') for i in PRODUCT_FEATURES.keys()})

    print(f"Собрано {len(data)} товаров с одной страницы")
    return data


# Получение характеристик товара
def product_info(product_url):
    html = fetch_webpage(product_url)
    if not html:
        return {}

    soup = BeautifulSoup(html, 'html.parser')
    details = {}

    try:
        price_div = soup.find('div', class_='v-price__item font-bold text-32')
        price = int(re.sub(r'\D', '', price_div.get_text(strip=True))) if price_div else None
    except (AttributeError, ValueError):
        price = None

    try:
        old_price_del = soup.find('del', class_='v-price__item text-grey-secondary text-24')
        old_price = int(re.sub(r'\D', '', old_price_del.get_text(strip=True))) if old_price_del else price
    except (AttributeError, ValueError):
        old_price = price

    details["Цена"] = price
    details["Старая цена"] = old_price

    # Характеристики
    for container in soup.select('div.v-detail-about dl div.break-inside-avoid-column'):
        dt = container.select_one('dt')
        dd = container.select_one('dd')
        if dt and dd:
            key = dt.get_text(strip=True)
            value = dd.get_text(strip=True)
            details[key] = value

            if key in category_values:
                category_values[key].add(value)

    # Заполняем недостающие значения
    for key in PRODUCT_FEATURES.keys():
        if key not in details:
            details[key] = 'Unknown'
        elif key in category_values and details[key] != 'Unknown':
            category_values[key].add(details[key])

    return details


# Создание JSON
def create_json(df):
    header = []

    for key in PRODUCT_FEATURES.keys():
        translated_name = TRANSLATION[key]
        attr_type = PRODUCT_FEATURES[key]
        feature_info = {"feature_name": translated_name}

        # Числовые поля
        if attr_type == 'NUMERIC':
            feature_info["type"] = "integer"

        # Категориальные поля
        else:
            if key in category_values and category_values[key]:
                unique_values = sorted([str(v) for v in category_values[key] if v != 'Unknown'])

                if len(unique_values) == 1:
                    unique_values = [unique_values[0]]

                if any("/" in val for val in unique_values):
                    feature_info["type"] = "multicategory"
                else:
                    feature_info["type"] = "category"

                feature_info["values"] = unique_values
            else:
                feature_info["type"] = "text"

        header.append(feature_info)

    # --- данные ---
    data = []
    for i, row in df.iterrows():
        record = {}
        for key in PRODUCT_FEATURES.keys():
            translated_name = TRANSLATION[key]
            value = row[key]

            if PRODUCT_FEATURES[key] == 'NUMERIC' and value != 'Unknown':
                try:
                    record[translated_name] = int(float(value))
                except (ValueError, TypeError):
                    record[translated_name] = value
            else:
                if isinstance(value, str) and "/" in value:
                    record[translated_name] = [v.strip() for v in value.split("/")]
                else:
                    record[translated_name] = value

        data.append(record)

    return {"header": header, "data": data}


# Сохранение в .tsv, .json и .arff
def raw_dataset(df):
    df.to_csv('data.tsv', sep='\t', index=False, encoding='utf-8')
    print("Сырые данные сохранены в data.tsv")

    json_data = create_json(df)

    def simplify_lists(obj):
        if isinstance(obj, list):
            if len(obj) == 1:
                return simplify_lists(obj[0])
            else:
                return [simplify_lists(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: simplify_lists(v) for k, v in obj.items()}
        else:
            return obj

    simplified_json = simplify_lists(json_data)

    def format_json(obj, indent=2, level=0):
        space = " " * (indent * level)
        next_space = " " * (indent * (level + 1))

        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = []
            for k, v in obj.items():
                items.append(f'\n{next_space}"{k}": {format_json(v, indent, level + 1)}')
            return "{" + ",".join(items) + f"\n{space}" + "}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"
            if len(obj) <= 5 and all(isinstance(x, (str, int, float)) and len(str(x)) < 20 for x in obj):
                inline = ", ".join(json.dumps(x, ensure_ascii=False) for x in obj)
                return f"[{inline}]"
            else:
                items = [f"\n{next_space}{format_json(x, indent, level + 1)}" for x in obj]
                return "[" + ",".join(items) + f"\n{space}" + "]"

        else:
            return json.dumps(obj, ensure_ascii=False)

    with open('data.json', 'w', encoding='utf-8') as f:
        f.write(format_json(simplified_json, indent=2))
        f.write("\n")

    print("Данные сохранены в data.json")

    save_arff(df)
    print("Данные сохранены в data.arff")


def save_arff(df):
    with open('data.arff', 'w', encoding='utf-8') as f:
        f.write('% Boots dataset from Thomas Munz\n')
        f.write('@RELATION boots\n\n')

        for key in PRODUCT_FEATURES.keys():
            attr_name = TRANSLATION[key]
            attr_type = PRODUCT_FEATURES[key]
            if attr_type == 'STRING' and key in category_values and category_values[key]:
                unique_values = sorted([str(val) for val in category_values[key] if val != 'Unknown'])
                if unique_values:
                    values_str = ', '.join([f'"{v}"' for v in unique_values])
                    f.write(f'@ATTRIBUTE {attr_name} {{{values_str}}}\n')
                else:
                    f.write(f'@ATTRIBUTE {attr_name} STRING\n')
            else:
                f.write(f'@ATTRIBUTE {attr_name} NUMERIC\n')

        f.write('\n@DATA\n')
        for _, row in df.iterrows():
            row_data = []
            for key in PRODUCT_FEATURES.keys():
                value = row[key]
                if pd.isna(value) or value == 'Unknown':
                    row_data.append('?')
                elif PRODUCT_FEATURES[key] == 'NUMERIC':
                    row_data.append(str(value))
                else:
                    row_data.append(f'"{str(value)}"')
            f.write(','.join(row_data) + '\n')


# Сохранение в .csv
def save_csv():
    df = pd.read_csv('data.tsv', sep='\t', encoding='utf-8')
    one_value_categories = [key for key in category_values.keys()
                            if len(category_values[key]) <= 1 and key in df.columns]
    df.drop(columns=one_value_categories, inplace=True, errors='ignore')

    for column in df.columns:
        if df[column].dtype == object:
            if not df[column].empty and df[column].nunique() > 0:
                most_common_value = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
                df[column] = df[column].replace('Unknown', most_common_value)

    categorical_columns = [key for key, value in PRODUCT_FEATURES.items()
                           if value == 'STRING' and key in df.columns and key != 'Бренд']

    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=False, dtype=int)

    numeric_columns = [key for key, value in PRODUCT_FEATURES.items()
                       if value == 'NUMERIC' and key in df.columns]

    if numeric_columns:
        for col in numeric_columns:
            if df[col].isna().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)

        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns]).round(8)

    df.to_csv('data.csv', sep=',', index=False, encoding='utf-8')
    print("Обработанные данные сохранены в data.csv")


def main():
    base_url = 'https://thomas-muenz.ru/men/catalog/shoes/boots/'
    all_boots = all_products(base_url, 9)

    if all_boots:
        df = pd.DataFrame(all_boots)
        raw_dataset(df)
        save_csv()
        print(f"Парсинг завершён! Собрано {len(all_boots)} товаров.")
    else:
        print("Не удалось собрать данные")


if __name__ == "__main__":
    main()
