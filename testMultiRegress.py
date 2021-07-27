import pandas as pd

import config
import numpy as np
import utility
import plot_utility
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


def performance_with_different_kalman_filter():
    pass


if __name__ == "__main__":
    names = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree", "MLP"]
    classifiers = [
        RandomForestRegressor(),
        KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
        KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
        DecisionTreeRegressor(random_state=0),
        MLPRegressor(random_state=1, max_iter=50000)
    ]

    X, y = utility.load_dataset_arff("datasetTrain0")

    lor = y[:, 0]
    lop = y[:, 1]
    lox, loy = utility.pol2cart(lor, lop)

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.50)

    error_x_df = pd.DataFrame()
    error_y_df = pd.DataFrame()
    for name, clf in zip(names, classifiers):
        print("fit " + name)
        clf.fit(x_train, y_train)
        Z = clf.predict(X)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        error_x_df.insert(len(error_x_df.columns), name, error_x)
        error_y_df.insert(len(error_y_df.columns), name, error_y)

        df = pd.DataFrame({
            'predicted x': lpx,
            'optimal x': lox,
            'predicted y': lpy,
            'optimal y': loy
        })

        fig, axes = plt.subplots(2, 1)
        fig.suptitle(name)
        df.iloc[:, :2].plot(ax=axes[0])
        df.iloc[:, 2:].plot(ax=axes[1])

    error_x_df.plot.box(title="Error x")
    error_y_df.plot.box(title="Error y")
