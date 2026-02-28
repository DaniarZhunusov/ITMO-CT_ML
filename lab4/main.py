import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score

# Методы отбора признаков
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # Фильтрующий
from sklearn.feature_selection import RFE  # Обёрточный
from sklearn.feature_selection import SelectFromModel  # Встроенный

# Классификаторы
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from clustering import KMeansClustering
from custom_selectors import LassoEmbeddedSelector, RFEWrappedSelector, ChiSquareFilterSelector


# Загрузка данных
df = pd.read_csv('SMS.tsv', sep='\t', encoding='utf-8')
print(f"Размер датасета: {df.shape}")
print(f"Классы:\n{df['class'].value_counts()}")

# Предобработка
X = df['text']
y = df['class'].map({'ham': 0, 'spam': 1})

# Векторизация текста
vectorizer = CountVectorizer(max_features=1000, stop_words='english',
                             min_df=5, max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)
feature_names = vectorizer.get_feature_names_out()

print(f"\nРазмерность после векторизации: {X_vectorized.shape}")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42, stratify=y
)


# 2. БИБЛИОТЕЧНАЯ РЕАЛИЗАЦИЯ МЕТОДОВ ОТБОРА ПРИЗНАКОВ

# 2.1 Встроенный метод (Случайный лес)

rf_embedded_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    max_features=30
)
rf_embedded_selector.fit(X_train, y_train)


# 2.2 Обёрточный метод RFE

wrapper_selector = RFE(
    estimator=SVC(kernel='linear', random_state=42),
    n_features_to_select=30,
    step=50  # Увеличиваем шаг для ускорения
)

wrapper_selector.fit(X_train, y_train)
X_train_wrapper = wrapper_selector.transform(X_train)
X_test_wrapper = wrapper_selector.transform(X_test)


# 2.3 Фильтрующий метод Взаимная информация (Mutual Information)
filter_mi = SelectKBest(
    score_func=mutual_info_classif,
    k=30
)
filter_mi.fit(X_train, y_train)


# 30 Признаков (Слов)

# Обучение
own_lasso = LassoEmbeddedSelector(alpha=0.02, max_iter=500).fit(X_train, y_train)
own_rfe = RFEWrappedSelector(LogisticRegression(), 30, 5).fit(X_train, y_train)
own_chi2 = ChiSquareFilterSelector(k=30).fit(X_train, y_train, verbose=False)

# Вывод
methods = [
    ("Own Lasso", own_lasso.selected_[:30]),
    ("Own RFE", np.where(own_rfe.support_)[0][:30]),
    ("Own Chi2", own_chi2.selected_[:30]),
    ("Lib RF", np.where(rf_embedded_selector.get_support())[0][:30]),
    ("Lib RFE (SVC)", np.where(wrapper_selector.get_support())[0][:30]),
    ("Lib MI", np.where(filter_mi.get_support())[0][:30])
]

for name, idx in methods:
    if len(idx) > 0:
        words = feature_names[idx]
        print(f"\n{name}: ({len(words)} слов)")
        print("-" * 50)

        # Выводим по 5 слов в строку для удобства чтения
        for i in range(0, len(words), 5):
            line_words = words[i:i + 5]
            line_nums = [f"{j + 1 + i:2d}" for j in range(len(line_words))]
            line_text = "  ".join([f"{num}. {word:15}" for num, word in zip(line_nums, line_words)])
            print(line_text)
    else:
        print(f"\n{name}: Нет отобранных признаков")

# 3. СРАВНЕНИЕ КЛАССИФИКАТОРОВ ДО И ПОСЛЕ ОТБОРА ПРИЗНАКОВ

# Определяем 3 классификатора
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Собираем все селекторы
selectors = [
    ('Own Lasso', own_lasso),
    ('Own RFE', own_rfe),
    ('Own Chi2', own_chi2),
    ('Lib RF', rf_embedded_selector),
    ('Lib RFE (SVC)', wrapper_selector),
    ('Lib MI', filter_mi)
]


# Функция для получения отобранных признаков
def get_selected_indices(selector):
    if hasattr(selector, 'get_support'):
        return np.where(selector.get_support())[0]
    elif hasattr(selector, 'selected_'):
        return selector.selected_
    elif hasattr(selector, 'support_'):
        return np.where(selector.support_)[0]
    return []


# Оцениваем ДО отбора (на всех признаках)
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ДО ОТБОРА ПРИЗНАКОВ")
print("=" * 60)

# Создаем "методы" до отбора - по одному для каждого селектора
methods_before = [name for name, _ in selectors]
accuracies_before = {clf_name: [] for clf_name in classifiers.keys()}

for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    # Заполняем одинаковыми значениями для всех "методов"
    for _ in range(len(methods_before)):
        accuracies_before[clf_name].append(accuracy)
    print(f"{clf_name}: Accuracy = {accuracy:.4f}")

# Оцениваем ПОСЛЕ отбора (для каждого метода)
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ПОСЛЕ ОТБОРА ПРИЗНАКОВ")
print("=" * 60)

accuracies_after = {clf_name: [] for clf_name in classifiers.keys()}
for selector_name, selector in selectors:
    indices = get_selected_indices(selector)[:30]  # Берем топ-30
    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices]

    print(f"\n{selector_name}:")
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_selected, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test_selected))
        accuracies_after[clf_name].append(accuracy)
        print(f"  {clf_name}: Accuracy = {accuracy:.4f}")

# График 1: До отбора признаков
plt.figure(figsize=(12, 6))
x_before = np.arange(len(methods_before))
width = 0.25
colors_before = ['blue', 'red', 'green']

for i, (clf_name, acc_list) in enumerate(accuracies_before.items()):
    plt.bar(x_before + i * width, acc_list, width, label=clf_name,
            color=colors_before[i], alpha=0.7, edgecolor='black')

    # Добавляем подписи значений
    for j, acc in enumerate(acc_list):
        plt.text(x_before[j] + i * width, acc + 0.003,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)

plt.title('Точность классификаторов ДО отбора признаков', fontsize=14, fontweight='bold')
plt.xlabel('Метод (для сравнения)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(x_before + width, methods_before, rotation=45, ha='right')
plt.ylim([min([min(acc) for acc in accuracies_before.values()]) - 0.05, 1.0])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# График 2: После отбора признаков
plt.figure(figsize=(12, 6))
x_after = np.arange(len(selectors))
width = 0.25
colors_after = ['blue', 'red', 'green']

for i, (clf_name, acc_list) in enumerate(accuracies_after.items()):
    plt.bar(x_after + i * width, acc_list, width, label=clf_name,
            color=colors_after[i], alpha=0.7, edgecolor='black')

    # Добавляем подписи значений
    for j, acc in enumerate(acc_list):
        plt.text(x_after[j] + i * width, acc + 0.003,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)

plt.title('Точность классификаторов ПОСЛЕ отбора признаков', fontsize=14, fontweight='bold')
plt.xlabel('Метод отбора признаков', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(x_after + width, [name for name, _ in selectors], rotation=45, ha='right')
plt.ylim([min([min(acc) for acc in accuracies_after.values()]) - 0.05, 1.0])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# График 3: Сравнение улучшений
plt.figure(figsize=(10, 6))

# Рассчитываем изменение accuracy
for i, (clf_name, acc_list) in enumerate(accuracies_after.items()):
    baseline = accuracies_before[clf_name][0]  # Берем первое значение (они все одинаковые)
    improvements = [(acc - baseline) * 100 for acc in acc_list]  # в процентах
    plt.plot([name for name, _ in selectors], improvements,
             'o-', label=clf_name, color=colors_after[i], linewidth=2, markersize=8)

    # Добавляем подписи значений
    for j, imp in enumerate(improvements):
        plt.text(j, imp + (0.3 if imp >= 0 else -0.3),
                 f'{imp:+.2f}%', ha='center', va='bottom' if imp >= 0 else 'top',
                 fontsize=9)

plt.title('Изменение Accuracy после отбора признаков\n(относительно всех признаков)', fontsize=14, fontweight='bold')
plt.xlabel('Метод отбора признаков', fontsize=12)
plt.ylabel('ΔAccuracy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Выбор лучшего метода
print("\n" + "=" * 60)
print("ВЫБОР МЕТОДА ОТБОРА ПРИЗНАКОВ")
print("=" * 60)

# Вычисляем средний accuracy для каждого метода
method_names = [name for name, _ in selectors]
avg_accuracies = []

for i, method_name in enumerate(method_names):
    method_acc = [accuracies_after[clf][i] for clf in classifiers.keys()]
    avg_accuracy = np.mean(method_acc)
    avg_accuracies.append(avg_accuracy)
    print(f"{method_name}: Средняя Accuracy = {avg_accuracy:.4f}")

# Выбираем лучший метод
best_idx = np.argmax(avg_accuracies)
best_method = selectors[best_idx][0]
best_selector = selectors[best_idx][1]

# Базовый результат (все признаки)
baseline_avg = np.mean([accuracies_before[clf][0] for clf in classifiers.keys()])

print(f"\n Выбран метод: {best_method}")
print(f"   • Средняя Accuracy: {avg_accuracies[best_idx]:.4f}")
print(f"   • До отбора признаков: {baseline_avg:.4f}")
print(f"   • Изменение: {(avg_accuracies[best_idx] - baseline_avg):+.4f}")

# Сохраняем выбранные признаки для дальнейшего использования
selected_indices = get_selected_indices(best_selector)[:30]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

print(f"   • Количество признаков: {len(selected_indices)}")
print(f"   • Примеры отобранных слов: {feature_names[selected_indices][:10]}")


# 5. КЛАСТЕРИЗАЦИЯ ДО И ПОСЛЕ ОТБОРА ПРИЗНАКОВ

print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ")
print("="*60)

# Применяем кластеризацию K-Means
kmeans_before = KMeansClustering(n_clusters=2, random_state=42)
kmeans_after = KMeansClustering(n_clusters=2, random_state=42)

# Кластеризация до отбора
clusters_before = kmeans_before.fit_predict(X_train.toarray())
# Кластеризация после отбора
X_train_selected_dense = X_train_selected.toarray()
clusters_after = kmeans_after.fit_predict(X_train_selected_dense)

# Оценка качества
silhouette_before = silhouette_score(X_train.toarray(), clusters_before)
silhouette_after = silhouette_score(X_train_selected_dense, clusters_after)
ari_before = adjusted_rand_score(y_train, clusters_before)
ari_after = adjusted_rand_score(y_train, clusters_after)

print(f"ДО отбора: Silhouette={silhouette_before:.4f}, ARI={ari_before:.4f}")
print(f"ПОСЛЕ отбора: Silhouette={silhouette_after:.4f}, ARI={ari_after:.4f}")

# 6. PCA И t-SNE ВИЗУАЛИЗАЦИЯ


# Берем подвыборку для t-SNE
sample_size = 500
indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_sample = X_train[indices].toarray()
X_selected_sample = X_train_selected_dense[indices]
y_sample = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
clusters_before_sample = clusters_before[indices]
clusters_after_sample = clusters_after[indices]

# PCA
pca_before = PCA(n_components=2, random_state=42)
X_pca_before = pca_before.fit_transform(X_sample)

pca_after = PCA(n_components=2, random_state=42)
X_pca_after = pca_after.fit_transform(X_selected_sample)

# t-SNE
tsne_before = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_before = tsne_before.fit_transform(X_sample)

tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_after = tsne_after.fit_transform(X_selected_sample)

# Визуализация
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# PCA - реальные классы
axes[0,0].scatter(X_pca_before[:,0], X_pca_before[:,1], c=y_sample, cmap='viridis', alpha=0.6)
axes[0,0].set_title('PCA: Реальные классы (до)')
axes[0,1].scatter(X_pca_after[:,0], X_pca_after[:,1], c=y_sample, cmap='viridis', alpha=0.6)
axes[0,1].set_title('PCA: Реальные классы (после)')

# PCA - кластеризация
axes[0,2].scatter(X_pca_before[:,0], X_pca_before[:,1], c=clusters_before_sample, cmap='plasma', alpha=0.6)
axes[0,2].set_title('PCA: Кластеризация (до)')
axes[0,3].scatter(X_pca_after[:,0], X_pca_after[:,1], c=clusters_after_sample, cmap='plasma', alpha=0.6)
axes[0,3].set_title('PCA: Кластеризация (после)')

# t-SNE - реальные классы
axes[1,0].scatter(X_tsne_before[:,0], X_tsne_before[:,1], c=y_sample, cmap='viridis', alpha=0.6)
axes[1,0].set_title('t-SNE: Реальные классы (до)')
axes[1,1].scatter(X_tsne_after[:,0], X_tsne_after[:,1], c=1-y_sample, cmap='viridis', alpha=0.6)
axes[1,1].set_title('t-SNE: Реальные классы (после)')

# t-SNE - кластеризация
axes[1,2].scatter(X_tsne_before[:,0], X_tsne_before[:,1], c=clusters_before_sample, cmap='plasma', alpha=0.6)
axes[1,2].set_title('t-SNE: Кластеризация (до)')
axes[1,3].scatter(X_tsne_after[:,0], X_tsne_after[:,1], c=1-clusters_after_sample, cmap='plasma', alpha=0.6)
axes[1,3].set_title('t-SNE: Кластеризация (после)')

plt.tight_layout()
plt.show()
