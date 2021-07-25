import config
import numpy as np
import utility
import plot_utility

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

if __name__ == "__main__":
    names = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree", "MLP"]
    classifiers = [
        RandomForestRegressor(),
        KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
        KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
        DecisionTreeRegressor(random_state=0),
        MLPRegressor(random_state=1, max_iter=50000)
    ]

    X, y = utility.load_dataset("datasetTrain0")

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.50)

    for name, clf in zip(names, classifiers):
        print("fit " + name)
        clf.fit(x_train, y_train)
        Z = clf.predict(X)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lor = y[:, 0]
        lop = y[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)
        lox, loy = utility.pol2cart(lor, lop)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        data = [
            {"error x": error_x},
            {"error y": error_y},
            {
                'Predicted': lpx,
                'Optimal': lox
            },
            {
                'Predicted': lpy,
                'Optimal': loy
            }
        ]

        plot_utility.plot(name, data, 2, 2)
