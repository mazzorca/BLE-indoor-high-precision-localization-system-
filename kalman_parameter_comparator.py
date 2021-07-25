"""
Script that compare different kalman filters
"""

import config
import numpy as np
import statistics
import pandas as pd

import utility
import data_extractor

# Experiment Set-up
INITIAL_R = 0.001
INITIAL_Q = 0.000001

FINAL_R = 0.1
FINAL_Q = 0.0001

NUM_R = 4
NUM_Q = 4

WINDOWS_SIZE = 50

STD_TH = 1.

if __name__ == "__main__":

    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cut = utility.get_index_taglio_reader(raws_time)

    experiment_dict = {
        'R': [],
        'Q': [],
        'T_RAISE': [],
        'STABILITY': []
    }

    selected_cut = 0
    kalman_filter_par = config.KALMAN_BASE
    for R in np.linspace(INITIAL_R, FINAL_R, NUM_R):
        kalman_filter_par['R'] = R
        for Q in np.linspace(INITIAL_Q, FINAL_Q, NUM_Q):
            kalman_filter_par['Q'] = Q
            kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
            chunks_reader = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)

            t_raise = []
            diff_mean = []
            for i, chunk in enumerate(chunks_reader):
                df_temp = chunk.copy()
                chunk['std'] = df_temp.rolling(WINDOWS_SIZE).std()
                chunk['mean'] = df_temp.rolling(WINDOWS_SIZE).mean()
                chunk['median'] = df_temp.rolling(WINDOWS_SIZE).median()

                index_std = chunk.loc[chunk['std'] < STD_TH].first_valid_index()
                start_index = index_cut[i][selected_cut]
                end_index = index_cut[i][selected_cut + 1] - start_index - 1

                if index_std is None:
                    index_std = end_index

                t_raise.append(raws_time[i][start_index + index_std] - raws_time[i][start_index])

                chunk = chunk.drop(chunk.index[list(range(index_std))])

                min_mean = chunk['mean'].min()
                max_mean = chunk['mean'].max()
                if end_index == index_std:
                    diff_mean.append(abs(max_mean))
                else:
                    diff_mean.append(max_mean - min_mean)

            experiment_dict['R'].append(R)
            experiment_dict['Q'].append(Q)
            experiment_dict['T_RAISE'].append(statistics.mean(t_raise))
            experiment_dict['STABILITY'].append(statistics.mean(diff_mean))

    experiment_df = pd.DataFrame(experiment_dict)
    experiment_df.set_index(['R', 'Q'])

    experiment_df.to_excel("kalman_parameter_comparison.xlsx")
