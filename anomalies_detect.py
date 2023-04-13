#!/usr/bin/env python


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import sys


from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression


def filter_ones(data):
    return data.WEFAC == 1


def filter_more_than_zero(data):
    return data.WEFAC > 0


def filter_all(data):
    return data.WEFAC >= 0


def filter_zeros(data):
    return data.WEFAC == 0


def highlight_points_on_lineplot(data, x, y, points, marker="o", color="red", ax=None):
    data = data.copy(deep=True)
    # Строим график линии
    sns.lineplot(x=x, y=y, data=data, ax=ax)
    # Добавляем маркеры для каждой выбранной точки
    data = data[points]
    #     sns.scatterplot(data = pd.DataFrame({'x':[1,2,3], 'y':[2,5,6]}), x='x', y='y', marker="o", color="red")
    for point in data[x]:
        sns.scatterplot(
            data=pd.DataFrame({"x": point, "y": data.loc[data[x] == point, y]}),
            x="x",
            y="y",
            marker=marker,
            color=color,
            ax=ax,
        )
        ax.set(xlabel=x, ylabel=y)
        ax.xaxis.get_label().set_fontsize(20)
        ax.yaxis.get_label().set_fontsize(20)
    # Отображаем график


#     plt.show()


def find_group_dbscan(arr, eps=0.5, min_samples=5):
    # Создаем объект DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # Преобразуем массив в двумерный формат
    X = arr.reshape(-1, 1)
    # Обучаем модель
    dbscan.fit(X)
    # Находим номера кластеров
    labels = dbscan.labels_ - np.min(dbscan.labels_)
    # Находим номер кластера, содержащего наибольшее количество точек
    largest_cluster = np.argmax(np.bincount(labels))
    # Находим индексы точек, которые не принадлежат этому кластеру
    outliers = labels != largest_cluster
    return outliers


def find_outliers_quartile(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


def find_outliers_lof(data, n=5):
    lof = LocalOutlierFactor(n_neighbors=n)
    outliers = lof.fit_predict(data.reshape(-1, 1))
    return outliers == -1


def find_trend_change_lr(data, threshold=0.4):
    indexes = np.full(len(data), False)
    x = np.arange(len(data)).reshape(-1, 1)
    prev_model = LinearRegression().fit(x[:2], data[:2])
    for i in range(2, len(data)):
        model = LinearRegression().fit(x[:i], data[:i])
        if np.abs(model.coef_ - prev_model.coef_) > threshold * np.abs(
            prev_model.coef_
        ):
            indexes[i - 1] = True
            prev_model = model
    return indexes


def find_trend_change_slide(data, window_size=5, threshold=0.5):
    ma1 = np.convolve(data, np.ones(window_size) / window_size, mode="same")
    ma2 = np.convolve(
        data, np.ones(2 * window_size + 1) / (2 * window_size + 1), mode="same"
    )
    diff = np.abs(ma1 - ma2)
    change_points = diff > threshold
    return change_points


def find_plateau(data):
    def change_false_to_true(array):
        for i in range(len(array) - 1):
            if array[i] == False and array[i + 1] == True:
                array[i] = True
        return array

    diffs = np.abs(np.diff(data))
    plateau_positions = [False] + list(diffs == 0)
    #     return change_false_to_true(plateau_positions)
    return plateau_positions


def find_zeros(data):
    plateau_positions = list(data == 0)
    return plateau_positions


def find_large_changes(arr, window_size=6, threshold=2):
    """
    Функция для поиска значений в массиве, когда изменение было больше в два раза,
    чем предыдущие 5 изменений

    Parameters:
    arr (numpy.ndarray): Входной массив значений

    Returns:
    numpy.ndarray: Массив индексов значений, когда изменение было больше в два раза
    """
    # Создаем скользящее окно размера 6
    window = np.zeros(window_size)
    # Заполняем первые 5 элементов окна разностями между значениями
    # массива (текущее значение - предыдущее значение)
    window[: window_size - 1] = np.diff(arr[:window_size])
    # Создаем пустой массив для хранения индексов аномальных значений
    indexes = np.full(len(data), False)
    # Проходим по всем оставшимся элементам массива
    for i in range(window_size, len(arr)):
        # Вычисляем разность между текущим и предыдущим значением
        diff = arr[i] - arr[i - 1]
        # Добавляем эту разность в окно
        window[-1] = diff
        # Вычисляем среднюю разность в окне (исключая последнее значение)
        mean = np.mean(window[:-1])
        # Проверяем, насколько текущая разность отличается от средней разности
        # Если отличие больше чем в два раза, считаем это значение аномалией
        print(mean, diff, arr[i], arr[i - 1])
        if np.abs(diff) >= np.abs(mean) * threshold:
            indexes[i] = True
        # Сдвигаем окно вправо
        window[:-1] = window[1:]
    # Преобразуем список аномальных индексов в массив и возвращаем его
    return np.array(indexes)


def slide_aggregate(
    data, window=30, method=find_group_dbscan, param_set=dict(), threshold=0.2
):
    cur_window = min(len(data), window)
    outliers_check = np.full((len(data) - cur_window + 1, len(data)), np.nan)
    for i in range(len(outliers_check)):
        predict = method(data[i : i + cur_window], **param_set)
        outliers_check[i][i : i + cur_window] = predict
    predicted_outliers = np.nansum(outliers_check, axis=0)
    total_predictions = np.count_nonzero(~np.isnan(outliers_check), axis=0)
    percentage = predicted_outliers / total_predictions
    return percentage > threshold


def preprocess(data):
    data = data.copy(deep=True)
    columns = ["WWPR", "WOPR", "WGPR", "WBHP", "WTHP", "WGIR", "WWIR"]
    for c in columns + ["WEFAC"]:
        if data[c].dtype == "O":
            data[c] = pd.Series(
                data[c].astype(str).apply(lambda x: x.replace(",", ".")).astype("float64")
            )
    data["WLPROD"] = (data["WWPR"] + data["WOPR"]) / (180 - data["WBHP"])
    data["WWPROD"] = (data["WWPR"]) / (180 - data["WBHP"])
    data["WOPROD"] = (data["WOPR"]) / (180 - data["WBHP"])
    data["GOR"] = (data["WGPR"]) / (data["WOPR"])
    return data

columns = ["WOPR", "WWPR", "WGPR", "WOPROD", "WLPROD", "GOR"]
functions = {
    "dbscan": find_group_dbscan,
    "quartiles": find_outliers_quartile,
    "LOF_method": find_outliers_lof,
    "sliding": find_trend_change_slide,
    "ensemble": None,
    "plateau": find_plateau,
    "zeros": find_zeros,
}
fun_names = {
    "dbscan": "DBSCAN",
    "quartiles": "Метод Квартилей",
    "LOF_method": "Локальный уровень выброса",
    "sliding": "Скользящее окно",
    "ensemble": "Ансамбль",
    "plateau": "Определение плато",
    "zeros": "Нули",
}
params = [
    dict(),
    dict(),
    {"n": 30},
    {"window_size": 20, "threshold": 0.4},
    None,
    dict(),
    dict(),
]



data_dir = sys.argv[1] #"./data/Фактические-синтетические данные"
images_dir = sys.argv[2] #'./images'
data_name = sys.argv[3] #"ГТМ_84.xlsx"

# print(sys.argv)

print(f"Working with {data_name}!")
data_file = os.path.join(data_dir, data_name)
data_file_to = os.path.join(data_dir, f"anomalies_{data_name.split('.')[0]}.csv")
data = pd.read_excel(data_file, decimal=".")
data = preprocess(data)
for well in np.unique(data.WELL):
    print(f"Working with well #{well}!")
    cur_data = (
        data[(data.WELL == well) & filter_all(data)]
        .fillna(0)[columns + ["DATE"]]
    )
    answers = dict()
    for fun_name, param in tqdm(list(zip(functions, params))):
        fun = functions[fun_name]
        TITLE = f"Метод: {fun_names[fun_name]}"
        sns.set(rc={"figure.figsize": (30, 15)})
        sns.set_theme(style="whitegrid")
        fig, axs = plt.subplots(nrows=2, ncols=3)
        fig.tight_layout(pad=5)
        for i, column in enumerate(columns):
            row = i // 3
            col = i % 3
            if fun_name=='ensemble':
                points = np.array(
                    answers[f"quartiles_{column}"].astype(int)
                    + answers[f"LOF_method_{column}"].astype(int)
                ) > 0  # half or more for anomaly
            else:
                points = slide_aggregate(
                    cur_data[column].to_numpy(), window=100, method=fun, param_set=param
                )
            answers[f"{fun_name}_{column}"] = np.array(points).astype(int)
            highlight_points_on_lineplot(
                cur_data, "DATE", column, points, ax=axs[row][col]
            )
        fig.suptitle(TITLE, fontsize=25)
        plt.savefig(
            os.path.join(images_dir, f"anomalies_{data_name.split('.')[0].replace(' ','_')}_{fun_name}_{well}.pdf")
        )
        plt.clf()
        print(fun_name, " done!")
    new_cols = []
    for column in columns:
        is_anomaly = (
            (
                answers[f"dbscan_{column}"]
                + answers[f"quartiles_{column}"]
                + answers[f"LOF_method_{column}"]
                + answers[f"sliding_{column}"]
            )
            > 2
        ) # half or more for anomaly
        is_plateau = answers[f"plateau_{column}"]
        is_zeros = answers[f"zeros_{column}"]
        cur_data[f"{column}_anomaly"] = is_anomaly.astype(int)
        cur_data[f"{column}_plateau"] = is_plateau
        cur_data[f"{column}_zero"] = is_zeros
        new_cols.append(f"{column}_anomaly")
        new_cols.append(f"{column}_plateau")
        new_cols.append(f"{column}_zero")
    for col in new_cols:
        data.loc[cur_data.index,[col]] = cur_data[col].to_numpy().astype(int)
data.to_csv(data_file_to, index=False)
print(f'Saved to {data_file_to}')