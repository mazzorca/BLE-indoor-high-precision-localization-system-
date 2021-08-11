"""
Script that compare different kalman filters
"""
import numpy as np
import pandas as pd

import config
import testMultiRegress
import utility
import data_extractor
import dataset_generator

# Experiment Set-up
INITIAL_R = 0.001
INITIAL_Q = 0.000001

FINAL_R = 0.1
FINAL_Q = 0.0001

NUM_R = 50
NUM_Q = 50

WINDOWS_SIZE = 50

RSSI_BAND = 2.


def get_settling_sample(kalman_chunks, raw_chunks):
    settling_sample = []
    for reader_num, chunks in enumerate(zip(kalman_chunks, raw_chunks)):
        kalman_chunks = chunks[0]
        raw_chunks = chunks[1]

        settling_sample.append([])
        for kalman_chunk, raw_chunk in zip(kalman_chunks, raw_chunks):
            df_temp = kalman_chunk.copy()
            kalman_chunk['std'] = df_temp.rolling(WINDOWS_SIZE).std()
            kalman_chunk['mean'] = df_temp.rolling(WINDOWS_SIZE).mean()
            kalman_chunk['median'] = df_temp.rolling(WINDOWS_SIZE).median()

            raw_mean = np.mean(raw_chunk['RSSI Value'])

            scaled_kalman_chunk = kalman_chunk['RSSI Value'].subtract(raw_mean)

            index_settling = 0
            for index, value in scaled_kalman_chunk.items():
                if value > RSSI_BAND or value < -RSSI_BAND:
                    index_settling = index
                if index > index_settling + config.MAX_T_ASS_SAMPLES:
                    break

            settling_sample[reader_num].append(index_settling)

    return settling_sample


def kalman_settling_sample_comparator(specific_kalman_filters=None, enable_regress_comparator=False):
    """

    :param specific_kalman_filters: list of kalman filters
    :param enable_regress_comparator: add to the df, the regressor error IQR and Median
    :return:
    """
    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cut = utility.get_index_start_and_end_position(raws_time)

    experiment_dict = {
        'R': [],
        'Q': [],
        'SETTLING_SAMPLE_AVG': [],
        'SETTLING_SAMPLE_MIN': [],
        'SETTLING_SAMPLE_MAX': []
    }

    if specific_kalman_filters is None:
        filters_dict = {
            'R': [],
            'Q': []
        }
        for R in np.linspace(INITIAL_R, FINAL_R, NUM_R):
            for Q in np.linspace(INITIAL_Q, FINAL_Q, NUM_Q):
                filters_dict['R'].append(R)
                filters_dict['Q'].append(Q)
        specific_kalman_filters = pd.DataFrame(filters_dict)

    # all_index_ass0 = {}
    kalman_filter_par = config.KALMAN_BASE
    for index, row in specific_kalman_filters.iterrows():
        R = row['R']
        Q = row['Q']
        kalman_filter_par['R'] = R
        kalman_filter_par['Q'] = Q

        experiment_dict['R'].append(R)
        experiment_dict['Q'].append(Q)

        print('R', R, 'Q', Q)

        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        kalman_chunks = utility.get_chunk(kalman_data, index_cut)
        raw_chunks = utility.get_chunk(raws_data, index_cut)

        settling_sample = get_settling_sample(kalman_chunks, raw_chunks)

        if enable_regress_comparator:
            data_r, data_c = dataset_generator.get_processed_data_from_a_kalman_data(kalman_data, raws_time, "2605r0")
            X, y = dataset_generator.generate_dataset_from_final_data(data_r, data_c)

            errors = testMultiRegress.performance_dataset(X, y)

            for key in errors.keys():
                utility.create_or_insert_in_list(experiment_dict, key, errors[key])

        # all_index_ass0[f'{R}R {Q}Q'] = t_raise[0]
        settling_sample_np = np.array(settling_sample)
        experiment_dict['SETTLING_SAMPLE_AVG'].append(np.mean(settling_sample_np))

        settling_sample_min_np = []
        settling_sample_max_np = []
        for i in range(settling_sample_np.shape[1]):
            settling_sample_min_np.append(np.min(settling_sample_np[:, i]))
            settling_sample_max_np.append(np.max(settling_sample_np[:, i]))
        experiment_dict['SETTLING_SAMPLE_MIN'].append(np.mean(settling_sample_min_np))
        experiment_dict['SETTLING_SAMPLE_MAX'].append(np.mean(settling_sample_max_np))

    # all_index_ass0_df = pd.DataFrame(all_index_ass0)
    experiment_df = pd.DataFrame(experiment_dict)
    experiment_df.set_index(['R', 'Q'])

    experiment_df.to_excel("kalman_parameter_comparison-settling_sample.xlsx")


def get_raws_data_runs():
    name_files = ["BLE2605r", "dati3105run0r", "dati3105run1r", "dati3105run2r"]

    raws_runs_data = []
    raws_runs_time = []
    for name in name_files:
        raws_data, raws_time = data_extractor.get_raw_rssi_csv(name)
        raws_runs_data.append(raws_data)
        raws_runs_time.append(raws_time)

    return raws_runs_data, raws_runs_time


def get_best_good_points(how_many, file_excel):
    df_total = pd.read_excel(file_excel)
    df_sorted = df_total.nlargest(how_many, ["Random forest", "Nearest Neighbors D", "Nearest Neighbors U"])
    return df_sorted


def get_kalman_good_point():
    cam_files = ["2605r0", "Cal3105run0", "Cal3105run1", "Cal3105run2"]

    raws_run, raws_time = get_raws_data_runs()

    experiment_dict = {
        'R': [],
        'Q': []
    }

    parameter_coverage = 0
    total_run = NUM_R * NUM_Q

    kalman_filter_par = config.KALMAN_BASE
    for R in np.linspace(INITIAL_R, FINAL_R, NUM_R):
        kalman_filter_par['R'] = R
        for Q in np.linspace(INITIAL_Q, FINAL_Q, NUM_Q):
            kalman_filter_par['Q'] = Q

            parameter_coverage += 1
            complete_percentage = (parameter_coverage * 100) / total_run
            print('Completed:', f'{complete_percentage}%')

            X_list = []
            y_list = []
            for raws_data, raw_time, cam_file in zip(raws_run, raws_time, cam_files):
                kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
                final_data_reader, final_data_cam = dataset_generator.get_processed_data_from_a_kalman_data(kalman_data,
                                                                                                            raw_time,
                                                                                                            cam_file)
                X_run, y_run = dataset_generator.generate_dataset_from_final_data(final_data_reader, final_data_cam)
                X_list.append(X_run)
                y_list.append(y_run)

            X, y = dataset_generator.concatenate_dataset(X_list, y_list)

            experiment_dict['R'].append(R)
            experiment_dict['Q'].append(Q)

            good_points = testMultiRegress.get_number_of_good_point(X, y)

            for regressor_name in good_points.keys():
                utility.create_or_insert_in_list(experiment_dict, regressor_name, good_points[regressor_name])

    experiment_df = pd.DataFrame(experiment_dict)
    experiment_df.set_index(['R', 'Q'])

    experiment_df.to_excel("kalman_parameter_comparator-good_points.xlsx")


if __name__ == "__main__":
    # df = get_best_good_points(20, "kalman_parameter_comparator-good_points.xlsx")
    # kalman_settling_sample_comparator(df[['Q', 'R']])
    get_kalman_good_point()