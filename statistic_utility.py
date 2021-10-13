import numpy as np
import pandas as pd

import utility


def get_ecdf_euclidean_df(optimal_points, predicted_points, ecdf_name):
    ecdf_dict = {}
    errors = utility.get_euclidean_distance(optimal_points, predicted_points)

    hist, bin_edges = np.histogram(errors, bins='auto', density=True)
    ecdf = np.cumsum(hist * np.diff(bin_edges))

    if 0 not in bin_edges:
        ecdf = np.insert(ecdf, 0, 0)
        bin_edges = np.insert(bin_edges, 0, 0)
    bin_edges = np.delete(bin_edges, -1)

    ecdf_dict[f'ecdf_{ecdf_name}'] = ecdf

    df = pd.DataFrame(ecdf_dict, index=bin_edges)

    return df


def get_ecdf_euclidean_df_resolve_auto(optimal_points, predicted_points, ecdf_name):
    ecdf_dict = {}
    errors = utility.get_euclidean_distance(optimal_points, predicted_points)

    errors = np.sort(errors)
    weigth = 1/errors.shape[0]

    x = [0]
    y = [0]
    cumsum = 0
    last_error = 0
    for error in errors:
        if error == last_error:
            del x[-1]
            del y[-1]

        last_error = error
        x.append(error)
        cumsum += weigth
        y.append(cumsum)

    ecdf_dict[f'ecdf_{ecdf_name}'] = y

    df = pd.DataFrame(ecdf_dict, index=x)

    return df


def get_ecdf_square_df(xo, yo, xp, yp, ecdf_name):
    ecdf_dict = {}

    errors = []
    for xoi, yoi, xpi, ypi in zip(xo, yo, xp, yp):
        dist_x = abs(xoi - xpi)
        dist_y = abs(yoi - ypi)

        if dist_x > dist_y:
            errors.append(dist_x)
        else:
            errors.append(dist_y)

    bins = [0, 1, 2, 3, 4, 5, 6]
    hist, bin_edges = np.histogram(errors, bins=bins, density=True)
    ecdf = np.cumsum(hist * np.diff(bin_edges))

    if 0 not in bin_edges:
        ecdf = np.insert(ecdf, 0, 0)
        bin_edges = np.insert(bin_edges, 0, 0)

    bin_edges = np.delete(bin_edges, -1)
    ecdf_dict[ecdf_name] = ecdf

    df = pd.DataFrame(ecdf_dict, index=bin_edges)

    return df


def meter_at_given_percentage(ecdf, percentage):
    ecdf_name = list(ecdf.columns.values)[0]
    percentage = percentage / 100
    ecdf_new = ecdf[ecdf[ecdf_name] >= percentage]

    return ecdf_new.index[0]


def percentage_at_given_meter(ecdf, meter):
    ecdf_name = list(ecdf.columns.values)[0]
    ecdf_new = ecdf[ecdf.index >= meter]

    return ecdf_new[ecdf_name][ecdf_new.index[0]]

