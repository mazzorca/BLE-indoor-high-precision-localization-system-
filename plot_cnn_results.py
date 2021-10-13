import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections

import Configuration.cnn_config as cnn_conf
import config
import plot_utility
import regressors_lib
import statistic_utility
import utility

import testMultiRegress
import dataset_generator

testing_dataset = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
# testing_dataset = ["dati3105run0r"]
# testing_dataset = ["dati3105run1r"]
# testing_dataset = ["dati3105run2r"]

"""
    plots =
        1: compare with regressor
"""
plots = 0
choise = 0

best_models_name_cnn = [
    "resnet50_kalman/20-0.01-32-15x15-10",
    "resnet50_nokalman/20-0.01-128-20x20-10",
    "ble_kalman/20-0.01-32-20x20-10",
    "ble_nokalman/20-0.01-32-25x25-10",
    "wifi_kalman/20-0.01-32-25x25-10",
    "wifi_nokalman/20-0.01-32-25x25-10",
    "rfid_kalman/20-0.01-32-20x20-10",
    "rfid_nokalman/20-0.01-32-25x25-10",
    "alexnet_kalman/15-0.01-32-20x20-10",
    "alexnet_nokalman/20-0.01-32-20x20-10"
]

best_models_name = [
    "resnet50_kalman/20-0.01-32-15x15-10",
    "resnet50_nokalman/20-0.01-128-20x20-10",
    "ble_kalman/20-0.01-32-20x20-10",
    "ble_nokalman/20-0.01-32-25x25-10",
    "wifi_kalman/20-0.01-32-25x25-10",
    "wifi_nokalman/20-0.01-32-25x25-10",
    "rfid_kalman/20-0.01-32-20x20-10",
    "rfid_nokalman/20-0.01-32-25x25-10",
    "alexnet_kalman/15-0.01-32-20x20-10",
    "alexnet_nokalman/20-0.01-32-20x20-10",
    "rnn_kalman/x",
    "rnn_nokalman/x"
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
    if type_ecdf.split("-")[0] == "0":
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


def compare_cnns_with_ecdf(experiment_list, models_names, what_type_of_ecdf=0, type_dists=None):
    """
    Compare the performance of different regressor with a run by ecdf
    :param models_names:
    :param experiment_list: dataset where to test the models
    :param what_type_of_ecdf:
        0: euclidean
        1: square
        2: compare 2 or more type of euclidean ecdf
        3: compare 2 or more type of square ecdf
    :param type_dists: Com'è calcolato l'errore
    :return: void
    """

    name_experiment = ' '.join(experiment_list)
    name_plot = f"{name_experiment} compare rnns"
    save_name = f"{name_experiment.replace(' ', '_')}_rnns.png"
    if what_type_of_ecdf == 0:
        if type_dists is None:
            type_dists = ["0-1"]

        ecdf_total = compare_cnns_with_ecdf_euclidean(models_names, experiment_list, type_dists)

        ecdf_total.plot.line(
            title=f"ECDF {name_plot}",
            xlabel="(m)",
            ylabel="Empirical cumulative distribution function",
            cmap="plasma"
        )

        utility.check_and_if_not_exists_create_folder(f"plots/ecdf_euclidean_{save_name}")
        plt.savefig(f"plots/ecdf_euclidean_{save_name}")

    if what_type_of_ecdf == 1:
        fig, ax = plt.subplots()
        ax.set_title(f"ECDF {name_plot}")

        if type_dists is None:
            type_dists = ["1-1"]

        ax = compare_cnns_with_ecdf_square(models_names, experiment_list, ax, type_dists)

        plt.legend(loc='lower right')
        plt.show()

        plt.savefig(f'plots/ecdf_square_{save_name}')


def compare_cnns_with_ecdf_euclidean(models_names, experiment_list, type_dists):
    ecdf_total = pd.DataFrame()

    for model_name in models_names:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]
        par = model_name.split("/")[1]

        for type_dist in type_dists:
            argmax_number = type_dist.split("-")[1]
            predicteds, optimals = get_results(res_folder, experiment_list, par, type_dist)

            name = f"{model} {kalman_or_not} {argmax_number}"
            ecdf_df = statistic_utility.get_ecdf_euclidean_df(optimals, predicteds, name)
            ecdf_total = pd.concat([ecdf_total, ecdf_df], axis=1)

    ecdf_total = ecdf_total.interpolate(method='linear')

    return ecdf_total


def compare_cnns_with_ecdf_square(models_names, experiment_list, ax, type_dists):
    for model_name in models_names:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]
        par = model_name.split("/")[1]

        for type_dist in type_dists:
            argmax_number = type_dist.split("-")[1]
            predicteds, optimals = get_results(res_folder, experiment_list, par, type_dist)

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

            name = f"{model} {kalman_or_not} {argmax_number}"
            ecdf_df = statistic_utility.get_ecdf_square_df(xo, yo, xp, yp, name)

            index = ecdf_df.index.tolist()
            ax.step(np.array(index), ecdf_df[name], label=name, where="post")

    ax.set_xlabel("squares")
    ax.set_ylabel("Empirical cumulative distribution function")

    return ax


def compare_with_regressors_euclidean(model_name, experiment_list, type_dist):
    ecdf_total_cnn = compare_cnns_with_ecdf_euclidean([model_name], experiment_list, [type_dist])

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


def compare_with_regressors_square(model_name, experiment_list, type_dist):
    fig, ax = plt.subplots()

    name_plot = ' '.join(experiment_list)
    ax.set_title(f"ECDF {name_plot}")

    ax = compare_cnns_with_ecdf_square([model_name], experiment_list, ax, [type_dist])

    dataset_tests = dataset_generator.dataset_tests[name_plot]
    train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
    test_dataset = dataset_generator.load_dataset_numpy_file(dataset_tests[0], dataset_tests[1])

    ax = testMultiRegress.compare_regressor_with_ecdf_square(train_dataset, test_dataset,
                                                             testMultiRegress.CLASSIFIERS_DICT, ax)

    plt.legend(loc='lower right')
    plt.show()

    plt.savefig(f'plots/ecdf_square_{name_plot}.png')


def get_tables_of_best_argument_euclidean(models_names, experiment_list, percentage, type_dists=None):
    if type_dists is None:
        type_dists = [f"0-{i}" for i in range(1, 19)]

    table_dict = {}
    for model_name in models_names:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]
        par = model_name.split("/")[1]

        name = f"{model} {kalman_or_not}"
        for type_dist in type_dists:
            argmax_number = type_dist.split("-")[1]
            predicteds, optimals = get_results(res_folder, experiment_list, par, type_dist)

            name_df = f"{name} {argmax_number}"

            ecdf_df = statistic_utility.get_ecdf_euclidean_df(optimals, predicteds, name_df)
            meters = statistic_utility.meter_at_given_percentage(ecdf_df, percentage)
            meters = float("{:.3f}".format(meters))
            utility.create_or_insert_in_list(table_dict, name, meters)

    table_df = pd.DataFrame(table_dict, index=type_dists)
    table_df.to_excel("table_best_argmax_cnn.xlsx")

    return table_df


def get_tables_of_best_argument_square(models_names, experiment_list, squares, type_dists=None):
    if type_dists is None:
        type_dists = [f"1-{i}" for i in range(1, 19)]

    table_dict = {}
    for model_name in models_names:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]
        par = model_name.split("/")[1]

        name = f"{model} {kalman_or_not}"
        for type_dist in type_dists:
            argmax_number = type_dist.split("-")[1]
            predicteds, optimals = get_results(res_folder, experiment_list, par, type_dist)

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

            name_df = f"{name} {argmax_number}"
            ecdf_df = statistic_utility.get_ecdf_square_df(xo, yo, xp, yp, name_df)
            percentage = statistic_utility.percentage_at_given_meter(ecdf_df, squares)
            utility.create_or_insert_in_list(table_dict, name, percentage)

    table_df = pd.DataFrame(table_dict, index=type_dists)
    table_df.to_excel("table_best_argmax_cnn.xlsx")

    return table_df


def compare_best_cnns_euclidean():
    df = get_tables_of_best_argument_euclidean(best_models_name_cnn, testing_dataset, 90)

    ecdf_total = pd.DataFrame()
    for model_name in best_models_name_cnn:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]

        name = f"{model} {kalman_or_not}"

        best_argmax_value = df[name].min()
        type_dist = df.loc[df[name] == best_argmax_value].index[0]
        best_argmax_index = type_dist.split("-")[1]

        ecdf_df = compare_cnns_with_ecdf_euclidean([model_name], testing_dataset, [f"0-{best_argmax_index}"])
        ecdf_total = pd.concat([ecdf_total, ecdf_df], axis=1)

    ecdf_total = ecdf_total.interpolate(method='linear')

    save_name = "ecdf_euclidean_compare_best_cnns.png"
    ecdf_total.plot.line(
        title=f"ECDF compare bests CNN",
        xlabel="(m)",
        ylabel="Empirical cumulative distribution function"
    )

    plt.savefig(f"plots/{save_name}")


def compare_best_cnns_square():
    df = get_tables_of_best_argument_euclidean(best_models_name_cnn, testing_dataset, 90)

    fig, ax = plt.subplots()
    ax.set_title("ECDF compare bests CNN")
    for model_name in best_models_name_cnn:
        res_folder = model_name.split("/")[0]
        model = res_folder.split("_")[0]
        kalman_or_not = res_folder.split("_")[1]

        name = f"{model} {kalman_or_not}"

        best_argmax_value = df[name].min()
        type_dist = df.loc[df[name] == best_argmax_value].index[0]
        best_argmax_index = type_dist.split("-")[1]

        ax = compare_cnns_with_ecdf_square([model_name], testing_dataset, ax, [f"1-{best_argmax_index}"])

    save_name = "ecdf_square_compare_best_cnns.png"

    plt.legend(loc='lower right')
    plt.show()

    plt.savefig(f'plots/ecdf_square_{save_name}.png')


def get_ecdf_regressors_euclidean(experiment_list, regressor_to_use=None):
    if regressor_to_use is None:
        regressor_to_use = testMultiRegress.CLASSIFIERS_DICT

    name_plot = ' '.join(experiment_list)
    dataset_tests = dataset_generator.dataset_tests[name_plot]
    train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
    test_dataset = dataset_generator.load_dataset_numpy_file(dataset_tests[0], dataset_tests[1])
    ecdf_total_regressor = testMultiRegress.compare_regressor_with_ecdf_euclidean(train_dataset, test_dataset,
                                                                                  regressor_to_use)
    return ecdf_total_regressor


def get_ecdf_regressors_square(experiment_list, ax, regressor_to_use=None):
    if regressor_to_use is None:
        regressor_to_use = testMultiRegress.CLASSIFIERS_DICT

    name_plot = ' '.join(experiment_list)
    dataset_tests = dataset_generator.dataset_tests[name_plot]
    train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
    test_dataset = dataset_generator.load_dataset_numpy_file(dataset_tests[0], dataset_tests[1])
    ax = testMultiRegress.compare_regressor_with_ecdf_square(train_dataset, test_dataset,
                                                             regressor_to_use, ax)
    return ax


def compare_all_euclidean(experiment_list, models_conf):
    plot_total_df = pd.DataFrame()

    name_experiment = ' '.join(experiment_list)
    name_plot = f"{name_experiment} total comparison"
    save_name = f"plots/ecdf_euclidean_{name_experiment.replace(' ', '_')}_total.png"

    for model_conf in models_conf:
        model_name = model_conf[0]
        model_type = f"0-{model_conf[1]}"
        ecdf_df = compare_cnns_with_ecdf_euclidean([model_name], experiment_list, [model_type])

        plot_total_df = pd.concat([plot_total_df, ecdf_df], axis=1)

    regressor_to_use = {
        "Nearest Neighbors U": testMultiRegress.CLASSIFIERS_DICT["Nearest Neighbors U"],
        "Nearest Neighbors D": testMultiRegress.CLASSIFIERS_DICT["Nearest Neighbors U"]
    }
    ecdf_total_regressor = get_ecdf_regressors_euclidean(experiment_list, regressor_to_use=regressor_to_use)

    plot_total_df = pd.concat([plot_total_df, ecdf_total_regressor], axis=1)

    plot_total_df = plot_total_df.interpolate(method='linear')

    plot_total_df.plot.line(
        title=f"ECDF {name_plot}",
        xlabel="(m)",
        ylabel="Empirical cumulative distribution function"
    )

    utility.check_and_if_not_exists_create_folder(save_name)
    plt.savefig(save_name)


def compare_all_square(experiment_list, models_conf):
    name_experiment = ' '.join(experiment_list)
    name_plot = f"{name_experiment} total comparison"
    save_name = f"ecdf_square_{name_experiment.replace(' ', '_')}_total.png"

    fig, ax = plt.subplots()
    ax.set_title(f"ECDF {name_plot}")

    for model_conf in models_conf:
        model_name = model_conf[0]
        model_type = f"1-{model_conf[1]}"

        ax = compare_cnns_with_ecdf_square([model_name], experiment_list, ax,
                                           type_dists=[model_type]
                                           )

    regressor_to_use = {
        "Nearest Neighbors U": testMultiRegress.CLASSIFIERS_DICT["Nearest Neighbors U"],
        "Nearest Neighbors D": testMultiRegress.CLASSIFIERS_DICT["Nearest Neighbors U"]
    }
    ax = get_ecdf_regressors_square(experiment_list, ax, regressor_to_use=regressor_to_use)

    plt.legend(loc='lower right')
    plt.show()

    plt.savefig(f'plots/ecdf_square_{save_name}.png')


def plot_points_on_table(model_name, experiment_list, type_dist):
    res_folder = model_name.split("/")[0]
    par = model_name.split("/")[1]
    predicteds_dl, optimals_dl = get_results(res_folder, experiment_list, par, "0-1")

    xo, yo = utility.get_square_number_array(optimals_dl[:, 0], optimals_dl[:, 1])
    optimal_square_dl = []
    for square_x, square_y in zip(xo, yo):
        square_number = square_y * 6 + square_x
        optimal_square_dl.append(square_number)

    xp, yp = utility.get_square_number_array(predicteds_dl[:, 0], predicteds_dl[:, 1])
    predicted_square_dl = []
    for square_x, square_y in zip(xp, yp):
        square_number = square_y * 6 + square_x
        predicted_square_dl.append(square_number)

    xo_dl = []
    yo_dl = []
    xp_dl = []
    yp_dl = []
    for p, o in zip(predicted_square_dl, optimal_square_dl):
        square_x, square_y = utility.get_squarex_and_squarey(o)
        xo_dl.append(square_x)
        yo_dl.append(square_y)

        square_x, square_y = utility.get_squarex_and_squarey(p)
        xp_dl.append(square_x)
        yp_dl.append(square_y)

    errors = []
    for xoi, yoi, xpi, ypi in zip(xo_dl, yo_dl, xp_dl, yp_dl):
        dist_x = abs(xoi - xpi)
        dist_y = abs(yoi - ypi)

        if dist_x > dist_y:
            errors.append(dist_x)
        else:
            errors.append(dist_y)

    colors_set = plt.get_cmap("viridis")(np.linspace(0, 1, 18))
    segmenents = []
    colors = []
    for i in range(len(errors)):
        p_predicted = np.array(predicteds_dl[i, 0], predicteds_dl[i, 1])
        p_optimal = np.array(optimals_dl[i, 0], optimals_dl[i, 0])
        dist = np.linalg.norm(p_predicted - p_optimal)

        if errors[i] >= 2 and dist >= 0.45:
            segmenent = [(optimals_dl[i, 0], optimals_dl[i, 1]), (predicteds_dl[i, 0], predicteds_dl[i, 1])]
            segmenents.append(segmenent)
            square_x, square_y = utility.get_square_number(optimals_dl[i, 0], optimals_dl[i, 1], config.SQUARES)

            color = colors_set[square_y * 6 + square_x]
            colors.append(color)

    lc = collections.LineCollection(segmenents, colors=colors, linewidths=1)
    fig, ax = plt.subplots()

    plt.xlim(0, 1.80)
    plt.ylim(0, 0.90)

    ax.add_collection(lc)

    ax = plot_utility.add_grid_meters(ax)

    ax.autoscale()
    ax.margins(0.1)
    name_experiment = ' '.join(experiment_list)
    ax.set_title(name_experiment)

    # errors = utility.get_euclidean_distance(optimals_dl, predicteds_dl)

    name_plot = ' '.join(experiment_list)
    dataset_tests = dataset_generator.dataset_tests[name_plot]
    train_dataset = dataset_generator.load_dataset_numpy_file("x_train", "y_train")
    x_train, y_train = train_dataset[0], train_dataset[1]
    test_dataset = dataset_generator.load_dataset_numpy_file(dataset_tests[0], dataset_tests[1])
    x_test, y_test = test_dataset[0], test_dataset[1]

    optimal_regr, predicted_regr = regressors_lib.get_regressor_optimal_and_predicted_points(
        regressors_lib.CLASSIFIERS_DICT["Nearest Neighbors D"],
        x_train, x_test, y_train, y_test)

    xo_regr, yo_regr = utility.get_square_number_array(optimal_regr[:, 0], optimal_regr[:, 1])
    xp_regr, yp_regr = utility.get_square_number_array(predicted_regr[:, 0], predicted_regr[:, 1])

    errors = []
    for xoi, yoi, xpi, ypi in zip(xo_regr, yo_regr, xp_regr, yp_regr):
        dist_x = abs(xoi - xpi)
        dist_y = abs(yoi - ypi)

        if dist_x > dist_y:
            errors.append(dist_x)
        else:
            errors.append(dist_y)

    colors_set = plt.get_cmap("viridis")(np.linspace(0, 1, 18))
    segmenents = []
    colors = []
    for i in range(len(errors)):
        p_predicted = np.array(predicted_regr[i, 0], predicted_regr[i, 1])
        p_optimal = np.array(optimal_regr[i, 0], optimal_regr[i, 0])
        dist = np.linalg.norm(p_predicted - p_optimal)
        if errors[i] >= 2 or dist > 0.45:
            segmenent = [(optimal_regr[i, 0], optimal_regr[i, 1]), (predicted_regr[i, 0], predicted_regr[i, 1])]
            segmenents.append(segmenent)
            square_x, square_y = utility.get_square_number(optimal_regr[i, 0], optimal_regr[i, 1], config.SQUARES)

            color = colors_set[square_y * 6 + square_x]
            colors.append(color)

    lc = collections.LineCollection(segmenents, colors=colors, linewidths=1)
    fig, ax = plt.subplots()

    plt.xlim(0, 1.80)
    plt.ylim(0, 0.90)

    ax.add_collection(lc)

    ax = plot_utility.add_grid_meters(ax)

    ax.autoscale()
    ax.margins(0.1)
    name_experiment = ' '.join(experiment_list)
    ax.set_title(name_experiment)


if __name__ == '__main__':
    if plots == 0:
        models_names = [
            "rnn_kalman/x",
            "rnn_nokalman/x"
        ]

        compare_cnns_with_ecdf(testing_dataset, models_names, what_type_of_ecdf=choise)

    if plots == 1:
        net = "resnet50_nokalman/20-0.01-128-20x20-10"
        if choise == 0:
            compare_with_regressors_euclidean(net, testing_dataset, "0-11")
        if choise == 1:
            compare_with_regressors_square(net, testing_dataset, "1-1")

    if plots == 2:
        models_names = [
            "resnet50_kalman/20-0.01-32-15x15-10"
        ]

        type_dists1 = [
            # "0-1",
            # "0-2",
            # "0-3",
            # "0-4",
            # "0-5",
            "0-6",
            "0-7",
            "0-8",
            "0-9",
            # "0-10",
            # "0-11",
            # "0-12",
            # "0-13",
            # "0-14",
            # "0-15",
            # "0-16",
            # "0-17",
            "0-18"
        ]

        type_dists_all = [f"0-{i}" for i in range(1, 19)]

        compare_cnns_with_ecdf(testing_dataset, models_names, what_type_of_ecdf=choise, type_dists=type_dists_all)

    if plots == 3:
        if choise == 0:
            compare_best_cnns_euclidean()
        if choise == 1:
            compare_best_cnns_square()

    if plots == 4:
        models_conf = [
            ["rnn_kalman/x", "1"],
            ["ble_kalman/20-0.01-32-20x20-10", "6"]
        ]

        if choise == 0:
            compare_all_euclidean(testing_dataset, models_conf)
        if choise == 1:
            compare_all_square(testing_dataset, models_conf)

    if plots == 5:
        model_name = "rnn_kalman/x"

        plot_points_on_table(model_name, testing_dataset, "0-1")

    if plots == 6:
        df = get_tables_of_best_argument_square(best_models_name_cnn, testing_dataset, 1)
