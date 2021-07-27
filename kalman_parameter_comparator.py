"""
Script that compare different kalman filters
"""
import numpy as np
import pandas as pd

import config
import utility
import data_extractor

# Experiment Set-up
INITIAL_R = 0.009
INITIAL_Q = 0.000005

FINAL_R = 0.011
FINAL_Q = 0.000015

NUM_R = 10
NUM_Q = 10

WINDOWS_SIZE = 50

RSSI_BAND = 2.

if __name__ == "__main__":

    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cut = utility.get_index_start_and_end_position(raws_time)

    experiment_dict = {
        'R': [],
        'Q': [],
        'T_RAISE_AVG': [],
        'T_RAISE_MIN': [],
        'T_RAISE_MAX': []
    }

    all_index_ass0 = {}
    kalman_filter_par = config.KALMAN_BASE
    for R in np.linspace(INITIAL_R, FINAL_R, NUM_R):
        kalman_filter_par['R'] = R
        for Q in np.linspace(INITIAL_Q, FINAL_Q, NUM_Q):
            kalman_filter_par['Q'] = Q
            kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
            kalman_chunks = utility.get_chunk(kalman_data, index_cut)
            raw_chunks = utility.get_chunk(raws_data, index_cut)

            t_raise = []
            for reader_num, chunks in enumerate(zip(kalman_chunks, raw_chunks)):
                kalman_chunks = chunks[0]
                raw_chunks = chunks[1]

                t_raise.append([])
                for j, chunk in enumerate(zip(kalman_chunks, raw_chunks)):
                    kalman_chunk = chunk[0]
                    raw_chunk = chunk[1]

                    df_temp = kalman_chunk.copy()
                    kalman_chunk['std'] = df_temp.rolling(WINDOWS_SIZE).std()
                    kalman_chunk['mean'] = df_temp.rolling(WINDOWS_SIZE).mean()
                    kalman_chunk['median'] = df_temp.rolling(WINDOWS_SIZE).median()

                    raw_mean = np.mean(raw_chunk['RSSI Value'])

                    scaled_kalman_chunk = kalman_chunk['RSSI Value'].subtract(raw_mean)

                    index_ass = 0
                    for index, value in scaled_kalman_chunk.items():
                        if value > RSSI_BAND or value < -RSSI_BAND:
                            index_ass = index

                    t_raise[reader_num].append(index_ass)

            all_index_ass0[f'{R}R {Q}Q'] = t_raise[0]
            experiment_dict['R'].append(R)
            experiment_dict['Q'].append(Q)

            t_raise_np = np.array(t_raise)
            experiment_dict['T_RAISE_AVG'].append(np.mean(t_raise_np))

            t_raise_min_np = []
            t_raise_max_np = []
            for i in range(t_raise_np.shape[1]):
                t_raise_min_np.append(np.min(t_raise_np[:, i]))
                t_raise_max_np.append(np.max(t_raise_np[:, i]))
            experiment_dict['T_RAISE_MIN'].append(np.mean(t_raise_min_np))
            experiment_dict['T_RAISE_MAX'].append(np.mean(t_raise_max_np))

    all_index_ass0_df = pd.DataFrame(all_index_ass0)
    experiment_df = pd.DataFrame(experiment_dict)
    experiment_df.set_index(['R', 'Q'])

    experiment_df.to_excel("kalman_parameter_comparison.xlsx")
