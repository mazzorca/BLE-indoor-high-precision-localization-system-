import numpy as np
import pandas as pd

import config
import data_converter
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

# Script config
WHAT_TO_OUTPUT = {
    'SETTLING_TIME': True,
    'GOOD_POINT': False
}

cam_files = [
    "2605r0",
    "Cal3105run0",
    "Cal3105run1",
    "Cal3105run2"
]

name_files = [
    "BLE2605r",
    "dati3105run0r",
    "dati3105run1r",
    "dati3105run2r"
]


def get_settling_sample(kalman_chunks, raw_chunks):
    """

    :param kalman_chunks:
    :param raw_chunks:
    :return:
    [
        [
            settling_index_chunk0,
            settling_index_chunk1,
            .
            .
            .
            settling_index_chunkN
        ],
        ..., []
    ]
    """
    settling_sample = []
    for reader_num, chunks in enumerate(zip(kalman_chunks, raw_chunks)):
        kalman_chunks_reader = chunks[0]
        raw_chunks_reader = chunks[1]

        settling_sample.append([])
        for kalman_chunk, raw_chunk in zip(kalman_chunks_reader, raw_chunks_reader):
            df_temp = kalman_chunk.copy()
            kalman_chunk['std'] = df_temp.rolling(WINDOWS_SIZE).std()
            kalman_chunk['mean'] = df_temp.rolling(WINDOWS_SIZE).mean()
            kalman_chunk['median'] = df_temp.rolling(WINDOWS_SIZE).median()

            raw_mean = np.mean(raw_chunk['RSSI Value'])
            scaled_kalman_chunk = kalman_chunk['RSSI Value'].subtract(raw_mean)

            settling_index = 0
            for index, value in scaled_kalman_chunk.items():
                if value > RSSI_BAND or value < -RSSI_BAND:
                    settling_index = index
                if index > settling_index + config.MAX_T_ASS_SAMPLES:
                    break

            settling_sample[reader_num].append(settling_index)

    return settling_sample


if __name__ == "__main__":
    raws_run, raws_time = data_extractor.get_raws_data_runs(name_files)

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

            experiment_dict['R'].append(R)
            experiment_dict['Q'].append(Q)

            parameter_coverage += 1
            complete_percentage = (parameter_coverage * 100) / total_run
            print('Completed:', f'{complete_percentage}%')

            for raws_data, raw_time, cam_file in zip(raws_run, raws_time, cam_files):
                kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)

                if WHAT_TO_OUTPUT["SETTLING_TIME"]:
                    experiment_dict['SETTLING_SAMPLE_AVG'] = []
                    experiment_dict['SETTLING_SAMPLE_MIN'] = []
                    experiment_dict['SETTLING_SAMPLE_MAX'] = []

                    index_cut = utility.get_index_start_and_end_position(raw_time)
                    kalman_chunks = utility.get_chunk(kalman_data, index_cut)
                    raw_chunks = utility.get_chunk(raws_data, index_cut)

                    settling_sample = get_settling_sample(kalman_chunks, raw_chunks)

                    settling_sample_np = np.array(settling_sample)
                    experiment_dict['SETTLING_SAMPLE_AVG'].append(np.mean(settling_sample_np))

                    settling_sample_min_np = []
                    settling_sample_max_np = []
                    for i in range(settling_sample_np.shape[1]):
                        settling_sample_min_np.append(np.min(settling_sample_np[:, i]))
                        settling_sample_max_np.append(np.max(settling_sample_np[:, i]))
                    experiment_dict['SETTLING_SAMPLE_MIN'].append(np.mean(settling_sample_min_np))
                    experiment_dict['SETTLING_SAMPLE_MAX'].append(np.mean(settling_sample_max_np))

                if WHAT_TO_OUTPUT["GOOD_POINT"]:
                    pass


    experiment_df = pd.DataFrame(experiment_dict)
    experiment_df.set_index(['R', 'Q'])

    experiment_df.to_excel("kalman_parameter_comparison.xlsx")
