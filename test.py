import pandas as pd

import utility
import dataset_generator
import numpy as np
from sklearn.model_selection import train_test_split

import config
import testMultiRegress
import visualizer
import kalman_parameter_comparator

if __name__ == "__main__":
    # dataset, dataset_time = utility.extract_and_apply_kalman_csv("BLE2605r")
    # reader_cut, cam_cut = dataset_generator.get_processed_data_from_a_kalman_data(dataset, dataset_time, "2605r0")
    # dataset_generator.generate_dataset_from_final_data(reader_cut, cam_cut)

    # X_a = []
    # y_a = []
    # for name_file, cam_file in zip(config.NAME_FILES, config.CAM_FILES):
    #     X_r, y_r = dataset_generator.generate_dataset_with_mean_and_std(name_file, cam_file)
    #     X_a.append(X_r)
    #     y_a.append(y_r)
    #
    # X, y = dataset_generator.concatenate_dataset(X_a, y_a)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.20)

    # df = kalman_parameter_comparator.get_best_good_points(10, "kpc-good_points.xlsx")
    # visualizer.specific_kalman_filter(df[['Q', 'R']])

    kalman_filters_dict = {
        'Q': [0.0001, 0.00001, 0.00001],
        'R': [0.001, 0.01, 1]
    }

    df = pd.DataFrame(kalman_filters_dict)
    visualizer.specific_kalman_filter_chunck(df, selected_cut=9)
