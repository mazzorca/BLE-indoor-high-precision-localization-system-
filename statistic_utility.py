import numpy as np
import pandas as pd

import utility


def get_ecdf_euclidean_df(optimal_points, predicted_points, ecdf_name):
    ecdf_dict = {}
    errors = utility.get_euclidean_distance(optimal_points, predicted_points)

    hist, bin_edges = np.histogram(errors, bins='auto', density=True)
    ecdf = np.cumsum(hist * np.diff(bin_edges))

    ecdf = np.insert(ecdf, 0, 0)
    bin_edges = np.insert(bin_edges, 0, 0)
    bin_edges = np.delete(bin_edges, -1)

    ecdf_dict[f'ecdf_{ecdf_name}'] = ecdf

    df = pd.DataFrame(ecdf_dict, index=bin_edges)

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
