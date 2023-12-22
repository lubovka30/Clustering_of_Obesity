import warnings

import random
import simpsom as sps
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import describe
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
import csv
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, LeakyReLU, Add, Activation, ZeroPadding2D, Dropout
from keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.initializers import glorot_uniform
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import homogeneity_score, silhouette_score, completeness_score, v_measure_score

from sklearn import cluster, datasets, mixture
from scipy.spatial.distance import cosine

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings(action="ignore")


# Функция генерации произвольного цвета
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color


# Функция визуализации кластеров и расчета центроидов
def plot_clusters(df_clust, labels, need_pca=True, name_alorithm=''):
    # Для визуализации кластеров многомерных объектов понизим размерность методом выделения главных компонент
    if need_pca:
        pca = PCA(2)
        pca.fit(df_clust)
        X_PCA = pca.transform(df_clust)
        x, y = X_PCA[:, 0], X_PCA[:, 1]
    else:
        x, y = df_clust[:, 0], df_clust[:, 1]

    # Каждому кластеру назначим свой цвет на графике
    clust = np.unique(labels)
    colors = {}
    if len(clust) == 3:
        colors[clust[0]] = 'red'
        colors[clust[1]] = 'blue'
        colors[clust[2]] = 'green'
    else:
        for i in range(len(clust)):
            colors[clust[i]] = generate_color()

    # Прорисовываем график
    df1 = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    groups = df1.groupby('label')
    centroids = {}

    fig, ax = plt.subplots(figsize=(10, 10))

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=4,
                color=colors[name], label='cluster ' + str(name), mec='none', zorder=-1)
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

        centroid = (sum(group.x) / len(group.x), sum(group.y) / len(group.y))
        centroids[name] = centroid
        ax.scatter(centroid[0], centroid[1], color='black', marker='X', s=200, zorder=1)

    #     for i in range(len(centroids)):
    #         ax.scatter(centroids[i][0], centroids[i][1], color='black', zorder=1)

    ax.legend()
    ax.set_title(name_alorithm + " Кластеры")
    plt.show()

    # Проверим метрики кластеризации
    silhouette = silhouette_score(df_clust, labels, metric='euclidean')
    homogeneity = homogeneity_score(labels_true=y, labels_pred=labels)
    completeness = completeness_score(labels_true=y, labels_pred=labels)
    v_measure = v_measure_score(labels_true=y, labels_pred=labels)

    print('silhouette = ', silhouette)
    print('homogeneity = ', homogeneity)
    print('completeness = ', completeness)
    print('v_measure = ', v_measure)

    dfc = pd.DataFrame(centroids)
    arr = []
    for i in range(len(dfc.columns)):
        arr.append(dfc[dfc.columns[i]].to_list())

    return arr


def get_centroid(x, y):
    return sum(x) / len(x), sum(y) / len(y)


# Начало работы
obesity_df = pd.read_csv("ObesityDataSet.csv")
# obesity_df.info()
# obesity_df.head()
# print(obesity_df.head(10))
# obesity_df.describe().to_csv("statistics.csv", index=True)

# for column in obesity_df.select_dtypes(include=['object']).columns:
#     print(obesity_df[column].value_counts())
#     print("\n")

# построение ящиков с усами
# Выбор числовых столбцов для графиков
# numerical_columns = obesity_df.select_dtypes(include=['float64', 'int64']).columns
# num_columns = len(numerical_columns)

# Расчет количества строк и столбцов
# cols = 4  # Задание количества столбцов
# rows = -(-num_columns // cols)

# Создание рисунка с графиком для каждого числового столбца
# fig = make_subplots(rows=rows, cols=cols, subplot_titles=numerical_columns)

# Добавление диаграммы к каждой ячейке
# for i, col in enumerate(numerical_columns, 1):
#     fig.add_trace(
#         go.Box(y=obesity_df[col], name=col),
#         row=(i-1)//cols + 1,
#         col=(i-1)%cols + 1
#     )

# Обновление рисунка
# fig.update_layout(
#     height=300 * rows,  # Настройка высоты в зависимости от количества строк
#     width=1000,
#     template='plotly_dark',
#     title='Box Plots of Numerical Variables',
#     title_x=0.5,
#     showlegend=False
# )

# Показать рисунок
# fig.show()

# Выбор только числовых столбцов для матричного графика
# numerical_columns = obesity_df.select_dtypes(include=['float64', 'int64']).columns
# reduced_df = obesity_df[numerical_columns]

# Создание матричного графика с помощью Plotly Express
# fig = px.scatter_matrix(reduced_df)

# Обновление рисунка
# fig.update_layout(
#     height=1200,
#     width=1200,
#     title='Pairplot of Numerical Variables in the Dataset',
#     title_x=0.5
# )

# Показать рисунок
# fig.show()

# Создание копии фрейма данных
encoded_df = obesity_df.copy()

# Инициализация LabelEncoder
le = LabelEncoder()

# Перебор столбцов и преобразование каждого категориального столбца в числовой с помощью кодирования меток
for column in encoded_df.select_dtypes(include=['object']).columns:
    encoded_df[column] = le.fit_transform(encoded_df[column])

# # Расчет корреляционной матрицы и округление ее до двух десятичных знаков
# corr_matrix = encoded_df.corr().round(2)
#
# # Создание тепловой карты с помощью Plotly Express
# fig = px.imshow(corr_matrix,
#                 text_auto=True,
#                 labels=dict(x="Variable", y="Variable", color="Correlation"),
#                 x=corr_matrix.columns,
#                 y=corr_matrix.columns,
#                 aspect="auto",
#                 color_continuous_scale='RdBu_r')
#
# # Обновление рисунка
# fig.update_layout(
#     title='Correlation Matrix of Variables',
#     title_x=0.5,
#     width=1000,
#     height=1000
# )
#
# # Показать рисунок
# fig.show()

# Зафиксируем воспроизводимость экспериментов
RANDOM_SEED = 21
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Применение кодирования меток для бинарных категориальных столбцов
binary_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for column in binary_columns:
    if column in encoded_df.columns:
        encoded_df[column] = le.fit_transform(encoded_df[column])

# Применение кодирование One-Hot для номинальных категориальных столбцов
nominal_columns = ['Gender', 'CAEC', 'CALC', 'MTRANS']
encoded_df = pd.get_dummies(encoded_df, columns=nominal_columns, drop_first=True)

# Проверка и кодирование 'NObeyesdad'
if 'NObeyesdad' in encoded_df.columns:
    encoded_df['NObeyesdad_encoded'] = le.fit_transform(encoded_df['NObeyesdad'])
    # Удаление оригинальной колонки "NObeyesdad".
    encoded_df.drop('NObeyesdad', axis=1, inplace=True)

# Вывод на экран несколько первых строк, чтобы проверить кодировку
# print(encoded_df.head())

print(encoded_df['NObeyesdad_encoded'].value_counts())

# # Определение целевой переменной
# X = encoded_df.drop(['NObeyesdad_encoded'], axis=1)  # Убрать только целевую переменную
# y = encoded_df['NObeyesdad_encoded']  # Целевая переменная
#
# # Разделение данных на обучающую и тестовую выборки
# # X, X_test = train_test_split(X, test_size=0.2, random_state=42)
# # X, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Выделим столбцы для применения кластеризации
# X_train = X[['Age', 'Weight', 'family_history_with_overweight',
#              'FAVC', 'NCP', 'CH2O', 'FAF', 'SMOKE', 'Gender_1', ]]
#
# # Проверим столбцы, которые в итоге есть у датасета
# print(X_train.columns)
#
# # Проверим корреляцию столбцов в сборном датасете
# correlation = X_train.corr()
# plt.figure(figsize=(16, 12))
# sns.heatmap(correlation, annot=True, cmap='coolwarm')
# plt.show()
#
# # Запомним количество признаков
# input_num = X_train.shape[1]
#
#
# # Классический алгоритм Machine Learning для кластеризации K-Means
#
# # Кластеризуем
# kmeans = KMeans(n_clusters=6, max_iter=600, random_state=RANDOM_SEED)
# kmeans.fit(X_train)
# labels1 = kmeans.labels_.astype(int)
#
# # Выведем на график кластеры и метрики кластеризации,
# # предварительно понизив размерность методом выделения главных компонент
# centroids1 = plot_clusters(X_train, labels1, name_alorithm='KMeans')
# print('centroids: ')
# print(centroids1)
#
# # Проверим распределение величин по каждому признаку для разных классов
# X_train['cl_kmeans'] = labels1
#
# # Для быстрой интерпретации результата выведем сравнение средних значений по каждому кластеру/величин
# for col in X_train.columns[:input_num]:
#     fig, ax = plt.subplots(1, 2, figsize=(7,3))
#     ax[0].set_title('Mean ' + col)
#     X_train.groupby(by = 'cl_kmeans').mean()[col].plot(ax=ax[0], kind='bar')
#     ax[1].set_title('Median ' + col)
#     X_train.groupby(by = 'cl_kmeans').median()[col].plot(ax=ax[1], kind='bar')
#     fig.show()
#
# # Проверим количественное распределение
# print(X_train['cl_kmeans'].value_counts())
#
#
# # СЕТЬ КОХОНЕНА
#
# # Создадим сеть Кохонена размером 20 на 20 выходных нейронов и активируем периодические граничные условия (PBC)
# somModel = sps.SOMNet(20, 20, X_train) #, PBC=True
#
# # Обучим сеть в течение 1000 эпох с шагом learning rate = 0.01
# somModel.train(start_learning_rate=0.01, epochs=1000)
#
# # Получим карту признаков сниженной размерности, спроектировав наш датасет на плоскость при помощи обученной сети
# map_ = np.array((somModel.project(X_train)))
#
# # Проверим размерность карты
# map_.shape
#
# # Визуализируем веса каждой ячейки карты
# somModel.nodes_graph(colnum=0)
#
# # Визуализируем расстояния каждой ячейки карты до ее соседей
# somModel.diff_graph(show=True)
#
# # Кластеризуем преобразованные данные
# kmeans = KMeans(n_clusters=6,max_iter=600,random_state=RANDOM_SEED)
# kmeans.fit(map_)
# labels2 = kmeans.labels_.astype(int)
#
# # Визуализируем кластеры
# centroids2 = plot_clusters(map_, labels2, need_pca=False)
# print('centroids:')
# print(centroids2)
#
# # Выведем информацию из разных кластеров, чтобы можно было интерпретировать результат
# X_train['cl_som'] = labels2
#
# for col in X_train.columns[:input_num]:
#     fig, ax = plt.subplots(1, 2, figsize=(7,3))
#     ax[0].set_title('Mean ' + col)
#     X_train.groupby(by = 'cl_som').mean()[col].plot(ax=ax[0], kind='bar')
#     ax[1].set_title('Median ' + col)
#     X_train.groupby(by = 'cl_som').median()[col].plot(ax=ax[1], kind='bar')
#     fig.show()
#
# # Проверим количественное распределение кластеров
# print(X_train.cl_som.value_counts())

