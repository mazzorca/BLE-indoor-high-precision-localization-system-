import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Configuration.cnn_config as cnn_conf
import config
import statistic_utility
import utility

import testMultiRegress
import dataset_generator

# testing_dataset = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
# testing_dataset = ["dati3105run0r"]
# testing_dataset = ["dati3105run1r"]
testing_dataset = ["dati3105run2r"]


best_models_name = [
    "resnet50_kalman/20-0.01-32-15x15-10",
    "resnet50_nokalman/20-0.01-128-20x20-10",
    "ble_kalman/20-0.01-32-20x20-10",
    "ble_nokalman/20-0.01-32-25x25-10"
]


def load_cnn_results(p_file, o_filee):
    with open(f'cnn_results/{p_file}.npy', 'rb') as f:
        predicted = np.load(f)

    with open(f'cnn_results/{o_filee}.npy', 'rb') as f:
        optimal = np.load(f)

    return predicted, optimal


def get_results(model_name, experiments_list, par, type_ecdf):
    predicteds = np.array([])
    optimals = np.array([])
    if type_ecdf == 0:
        predicteds = np.array([[], []])
        predicteds = predicteds.transpose()

        optimals = np.array([[], []])
        optimals = optimals.transpose()
    for experiment in experiments_list:
        predicted, optimal = load_cnn_results(f'{model_name}/{type_ecdf}.{par}-{experiment}_p',
                                              f'{model_name}/{type_ecdf}.{par}-{experiment}_o')
        predicteds = np.concatenate([predicteds, predicted])
        optimals = np.concatenate([optimals, optimal])

    return predicteds, optimals


def compare_cnns_with_ecdf(experiment_list, models_names, what_type_of_ecdf=0):
    """
    Compare the performance of different regressor with a run by ecdf
    :param models_names:
    :param experiment_list: dataset where to test the models
    :param what_type_of_ecdf:
        0: euclidean
        1: square
        2: compare 2 type of square ecdf
    :return: void
    """

    name_plot = ' '.join(experiment_list)
    name_plot = f"{name_plot}_resnet50"
    if what_type_of_ecdf == 0:
        ecdf_total = compare_cnns_with_ecdf_euclidean(models_names, experiment_list)

        ecdf_total.plot.line(
            title=f"ECDF {name_plot}",
            xlabel="(m)",
            ylabel="Empirical cumulative distribution function"
        )

        plt.savefig(f'plots/ecdf_euclidean_{name_plot}.png')

    if what_type_of_ecdf == 1:
        fig, ax = plt.subplots()
        ax.set_title(f"ECDF {name_plot}")

        ax = compare_cnns_with_ecdf_square(models_names, experiment_list, ax, 1)

        plt.legend(loc='lower right')
        plt.show()

        plt.savefig(f'plots/ecdf_square_{name_plot}.png')

    if what_type_of_ecdf == 2:
        fig, ax = plt.subplots()
        ax.set_title(f"ECDF {name_plot}")

        ax = compare_cnns_with_ecdf_square(models_names, experiment_list, ax, "1-0")
        ax = compare_cnns_with_ecdf_square(models_names, experiment_list, ax, "1-1")

        plt.legend(loc='lower right')
        plt.show()

        plt.savefig(f'plots/ecdf_square_{name_plot}.png')


def compare_cnns_with_ecdf_euclidean(models_names, experiment_list):
    ecdf_total = pd.DataFrame()
    for model_name in models_names:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]
        par = model_name.split("/")[1]

        predicteds, optimals = get_results(res_folder, experiment_list, par, 0)

        name = f"{model} {kalman_or_not}"
        ecdf_df = statistic_utility.get_ecdf_euclidean_df(optimals, predicteds, name)
        ecdf_total = pd.concat([ecdf_total, ecdf_df], axis=1)

    ecdf_total = ecdf_total.interpolate(method='linear')

    return ecdf_total


def compare_cnns_with_ecdf_square(models_names, experiment_list, ax, type_ecdf):
    for model_name in models_names:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]
        par = model_name.split("/")[1]

        predicteds, optimals = get_results(res_folder, experiment_list, par, type_ecdf)

        xo = []
        yo = []
        xp = []
        yp = []
        for p, o in zip(predicteds, optimals):
            square_x, square_y = utility.get_squarex_and_squarey(o)
            xo.append(square_x)
            yo.append(square_y)

            square_x, square_y = utility.get_squarex_and_squarey(p)
            xp.append(square_x)
            yp.append(square_y)

        name = f"{model} {kalman_or_not}"
        if type_ecdf == "1-1":
            name = f"{name} more"

        if type_ecdf == "1-0":
            name = f"{name} singol"

        ecdf_df = statistic_utility.get_ecdf_square_df(xo, yo, xp, yp, name)

        index = ecdf_df.index.tolist()
        ax.step(np.array(index), ecdf_df[name], label=name, where="post")

    ax.set_xlabel("squares")
    ax.set_ylabel("Empirical cumulative distribution function")

    return ax


def compare_with_regressors_euclidean(model_name, experiment_list):
    ecdf_total_cnn = compare_cnns_with_ecdf_euclidean([model_name], experiment_list)

    name_plot = ' '.join(experiment_list)
    dataset_tests = dataset_generator.dataset_tests[name_plot]
    train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
    test_dataset = dataset_generator.load_dataset_numpy_file(dataset_tests[0], dataset_tests[1])

    ecdf_total_regressor = testMultiRegress.compare_regressor_with_ecdf_euclidean(train_dataset, test_dataset,
                                                                                  testMultiRegress.CLASSIFIERS_DICT)

    ecdf_total = pd.concat([ecdf_total_cnn, ecdf_total_regressor], axis=1)

    ecdf_total = ecdf_total.interpolate(method='linear')

    ecdf_total.plot.line(
        title=f"ECDF {name_plot}",
        xlabel="(m)",
        ylabel="Empirical cumulative distribution function"
    )

    plt.savefig(f'plots/ecdf_euclidean_{name_plot}.png')


def compare_with_regressors_square(model_name, experiment_list):
    fig, ax = plt.subplots()

    name_plot = ' '.join(experiment_list)
    ax.set_title(f"ECDF {name_plot}")

    ax = compare_cnns_with_ecdf_square([model_name], experiment_list, ax)

    dataset_tests = dataset_generator.dataset_tests[name_plot]
    train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
    test_dataset = dataset_generator.load_dataset_numpy_file(dataset_tests[0], dataset_tests[1])

    ax = testMultiRegress.compare_regressor_with_ecdf_square(train_dataset, test_dataset,
                                                             testMultiRegress.CLASSIFIERS_DICT, ax)

    plt.legend(loc='lower right')
    plt.show()

    plt.savefig(f'plots/ecdf_square_{name_plot}.png')


if __name__ == '__main__':
    plots = 0
    choise = 0

    if plots == 0:
        models_names = [
            "resnet50_kalman/20-0.01-32-15x15-10",
            "resnet50_nokalman/20-0.01-128-20x20-10"
        ]

        compare_cnns_with_ecdf(testing_dataset, models_names, what_type_of_ecdf=choise)

    if plots == 1:
        net = "resnet50_kalman/20-0.01-32-25x25-10"
        if choise == 0:
            compare_with_regressors_euclidean(net, testing_dataset)
        if choise == 1:
            compare_with_regressors_square(net, testing_dataset)
