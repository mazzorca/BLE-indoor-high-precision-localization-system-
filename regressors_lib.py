from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections

import config
import plot_utility
import utility
import dataset_generator

CLASSIFIERS_DICT = {
    "Random forest": RandomForestRegressor(),
    "Nearest Neighbors U": KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
    "Nearest Neighbors D": KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
    "Decision Tree": DecisionTreeRegressor(random_state=0)
}


def get_kNN_dict(max_k, stride, weights):
    kNN_dict = {}
    Ks = np.arange(0, max_k, stride)

    if 1 not in Ks:
        Ks[0] = 1

    for k in Ks:
        kNN_dict[f'kNN {weights}  k={k}'] = KNeighborsRegressor(k, weights=weights)

    return kNN_dict


def get_predicted_points(regressor, x_test):
    Z = regressor.predict(x_test)

    lpr = Z[:, 0]
    lpp = Z[:, 1]
    lpx, lpy = utility.pol2cart(lpr, lpp)

    return lpx, lpy


def get_optimal_points(y_test):
    lor = y_test[:, 0]
    lop = y_test[:, 1]
    lox, loy = utility.pol2cart(lor, lop)

    return lox, loy


def get_regressor_optimal_and_predicted_points(regressor, x_train, x_test, y_train, y_test):
    lox, loy = get_optimal_points(y_test)
    optimal_points = np.column_stack([lox, loy])

    regressor.fit(x_train, y_train)
    lpx, lpy = get_predicted_points(regressor, x_test)
    predicted_points = np.column_stack([lpx, lpy])

    return optimal_points, predicted_points


def get_regressor_predicted_points(regressor, x_train, x_test, y_train):
    regressor.fit(x_train, y_train)
    lpx, lpy = get_predicted_points(regressor, x_test)

    return lpx, lpy


def plot_optimal_and_predicted_points_per_dataset(regressor_name):
    name_files_reader = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    name_files_cam = ["Cal3105run0", "Cal3105run1", "Cal3105run2"]

    x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")

    for name_file_reader, name_file_cam in zip(name_files_reader, name_files_cam):
        x_test, y_test = dataset_generator.generate_dataset([name_file_reader], [name_file_cam],
                                                            dataset_generator.generate_dataset_base)

        optimal_points, predicted_points = get_regressor_optimal_and_predicted_points(CLASSIFIERS_DICT[regressor_name],
                                                                                      x_train, x_test, y_train, y_test)
        errors = utility.get_euclidean_distance(optimal_points, predicted_points)

        colors_set = plt.get_cmap("viridis")(np.linspace(0, 1, optimal_points.shape[0]))
        segmenents = []
        colors = []
        for i in range(optimal_points.shape[0]):
            segmenent = [(optimal_points[i, 0], optimal_points[i, 1]), (predicted_points[i, 0], predicted_points[i, 1])]
            segmenents.append(segmenent)
            color = colors_set[i]
            colors.append(color)

        lc = collections.LineCollection(segmenents, colors=colors, linewidths=1)
        fig, ax = plt.subplots()

        plt.xlim(0, 1.80)
        plt.ylim(0, 0.90)

        ax.add_collection(lc)

        ax = plot_utility.add_grid_meters(ax)

        ax.autoscale()
        ax.margins(0.1)
        ax.set_title(name_file_reader)

        plt.savefig(f"plots/errors_{name_file_reader}.png")

    plt.show()


if __name__ == '__main__':
    plot_optimal_and_predicted_points_per_dataset("Nearest Neighbors D")
