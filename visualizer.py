"""
This file contains the functions to visualize data
"""
import config

import numpy as np
import pandas as pd

import utility
import data_extractor


def plot_rssi(data):
    utility.equalize_data_with_nan(data)
    rssi_dict = {
        "Reader0": data[0],
        "Reader1": data[1],
        "Reader2": data[2],
        "Reader3": data[3],
        "Reader4": data[4]
    }
    rssi_df = pd.DataFrame(rssi_dict)

    rssi_df.plot.line()
    rssi_df.plot.line(subplots=True)


if __name__ == "__main__":
    if False:
        kalman_rssi, time_Reader = utility.extract_and_apply_kalman_csv("BLE2605r")
        index_taglio_reader = utility.get_index_taglio_reader(time_Reader)
        plot_rssi(kalman_rssi)

    if True:
        selected_cut = 0
        raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
        kalman_filter_par = config.KALMAN_BASE
        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        index_cut = utility.get_index_taglio_reader(raws_time)
        chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
        for i, chunk in enumerate(chunks):
            window_size = int((index_cut[i][selected_cut + 1] - index_cut[i][selected_cut]) / 10)
            df_temp = chunk.copy()
            chunk['std'] = df_temp.rolling(window_size).std()
            chunk['mean'] = df_temp.rolling(window_size).mean()
            chunk['median'] = df_temp.rolling(window_size).median()

            chunk.plot.line(subplots=True)