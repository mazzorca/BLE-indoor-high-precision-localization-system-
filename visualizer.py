"""
This file contains the functions to visualize data
"""
import config

import numpy as np
import pandas as pd

import utility
import data_extractor


def plot_kalman_rssi():
    kalman_rssi, time_Reader = utility.extract_and_apply_kalman_csv("BLE2605r")
    rssi_dict = {
        "Reader0": kalman_rssi[0],
        "Reader1": kalman_rssi[1],
        "Reader2": kalman_rssi[2],
        "Reader3": kalman_rssi[3],
        "Reader4": kalman_rssi[4]
    }
    rssi_df = pd.DataFrame(rssi_dict)

    rssi_df.plot.line()
    rssi_df.plot.line(subplots=True)


def plot_reader_rssi_stats():
    selected_cut = 0
    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    kalman_filter_par = config.KALMAN_BASE
    utility.change_Q_R(kalman_filter_par, 0.010111111, 0.000001)
    kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
    index_cut = utility.get_index_start_and_end_position(raws_time)
    chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
    for i, chunk in enumerate(chunks):
        window_size = 50
        df_temp = chunk.copy()
        chunk['std'] = df_temp.rolling(window_size).std()
        chunk['mean'] = df_temp.rolling(window_size).mean()
        chunk['median'] = df_temp.rolling(window_size).median()

        chunk.plot.line(subplots=True)


def plot_raw_and_kalman_rssi():
    raws_data, _ = data_extractor.get_raw_rssi_csv("BLE2605r")
    kalman_filter_par = config.KALMAN_BASE
    kalman_data1 = utility.apply_kalman_filter(raws_data, kalman_filter_par)

    plot_dict = {
        "RAW_RSSI": raws_data[0],
        "KALMAN_RSSI_MIN": kalman_data1[0]
    }
    df = pd.DataFrame(plot_dict)

    # df.plot.line(subplots=True)
    df.plot.line()


def different_kalman_filter():
    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cuts = utility.get_index_start_and_end_position(raws_time)
    raw_chunks = utility.get_chunk(raws_data, index_cuts)

    plot_dict = {}

    raw_means = []
    for raw_chunks_reader, index_cut_reader in zip(raw_chunks, index_cuts):
        raw_mean = []
        for raw_chunk, start, end in zip(raw_chunks_reader, index_cut_reader['start'], index_cut_reader['end']):
            raw_rssi_mean = np.mean(raw_chunk['RSSI Value'])
            raw_mean.extend([raw_rssi_mean] * (end - start + 1))
        raw_means.append(raw_mean)

    plot_dict['RSSI MEAN'] = raw_means[0]
    plot_dict['LIMIT UP'] = [elem + 2. for elem in raw_means[0]]
    plot_dict['LIMIT DOWN'] = [elem - 2. for elem in raw_means[0]]
    kalman_filter_par = config.KALMAN_BASE
    R = 0.011
    for Q in np.linspace(0.000005, 0.000015, 10):
        utility.change_Q_R(kalman_filter_par, R, Q)
        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        plot_dict[f"Q:{Q}"] = kalman_data[0]

    df = pd.DataFrame(plot_dict)

    # df.plot.line(subplots=True)
    df.plot.line()


if __name__ == "__main__":
    visualize = 3

    if visualize == 0:
        plot_kalman_rssi()

    if visualize == 1:
        plot_reader_rssi_stats()

    if visualize == 2:
        plot_raw_and_kalman_rssi()

    if visualize == 3:
        different_kalman_filter()
