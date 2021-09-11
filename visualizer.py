"""
This file contains the functions to visualize data
"""
import matplotlib.pyplot as plt

import config

import numpy as np
import pandas as pd

import utility
import data_extractor
import dataset_generator


def plot_kalman_rssi():
    kalman_rssi_to_be_equalized, time_Reader = utility.extract_and_apply_kalman_csv("BLE2605r")
    kalman_rssi = utility.equalize_data_with_nan(kalman_rssi_to_be_equalized)
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
    selected_cut = 2
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
        "RAW_RSSI_0": raws_data[0],
        "KALMAN_RSSI_0": kalman_data1[0]
    }
    df = pd.DataFrame(plot_dict)

    # df.plot.line(subplots=True)
    df.plot.line(
        title="Raw and filtered RSSI",
        xlabel="Sample",
        ylabel="RSSI(db)"
    )

    plt.savefig("plots/raw_and_kalman_rssi.png")


def different_kalman_filter():
    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cuts = utility.get_index_start_and_end_position(raws_time)
    raw_chunks = utility.get_chunk(raws_data, index_cuts)

    plot_dict = {}

    raw_means, bounds_up, bounds_down = get_raws_means_and_bounds_for_plot(index_cuts, raw_chunks)

    plot_dict['RSSI MEAN'] = raw_means[0]
    plot_dict['LIMIT UP'] = bounds_up[0]
    plot_dict['LIMIT DOWN'] = bounds_down[0]

    kalman_filter_par = config.KALMAN_BASE
    kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
    plot_dict['REFERENCE'] = kalman_data[0]

    R = 0.025
    # Q = 0.00001
    for Q in np.linspace(.0001, .000001, 11):
        utility.change_Q_R(kalman_filter_par, R, Q)
        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        plot_dict[f"Q:{Q}"] = kalman_data[0]
        # plot_dict[f"R:{R}"] = kalman_data[0]

    df = pd.DataFrame(plot_dict)

    # df.plot.line(subplots=True)
    df.plot.line()


def specific_kalman_filter(kalman_filters=None):
    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cuts = utility.get_index_start_and_end_position(raws_time)
    raw_chunks = utility.get_chunk(raws_data, index_cuts)

    plot_dict = {}

    raw_means, bounds_up, bounds_down = get_raws_means_and_bounds_for_plot(index_cuts, raw_chunks)

    plot_dict['RSSI MEAN'] = raw_means[0]
    plot_dict['LIMIT UP'] = bounds_up[0]
    plot_dict['LIMIT DOWN'] = bounds_down[0]

    kalman_filter_par = config.KALMAN_BASE
    kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
    plot_dict['REFERENCE'] = kalman_data[0]

    if kalman_filters is not None:
        for index, row in kalman_filters.iterrows():
            R = row['R']
            kalman_filter_par['R'] = R
            Q = row['Q']
            kalman_filter_par['Q'] = Q
            kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
            plot_dict[f'{index} - {R}R {Q}Q'] = kalman_data[0]
    else:
        kalman_filter_par = config.KALMAN_1
        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        plot_dict['SELECTED'] = kalman_data[0]

    df = pd.DataFrame(plot_dict)

    # df.plot.line(subplots=True)
    df.plot.line()


def specific_kalman_filter_chunck(kalman_filters=None, selected_cut=0):
    raws_data, raws_time = data_extractor.get_raw_rssi_csv("BLE2605r")
    index_cut = utility.get_index_start_and_end_position(raws_time)
    raw_chunks = utility.get_chunk(raws_data, index_cut)

    plot_dict = {}

    raw_means, bounds_up, bounds_down = get_raws_means_and_bounds_for_plot(index_cut, raw_chunks)

    kalman_filter_par = config.KALMAN_BASE

    if kalman_filters is not None:
        for index, row in kalman_filters.iterrows():
            R = row['R']
            kalman_filter_par['R'] = R
            Q = row['Q']
            kalman_filter_par['Q'] = Q
            kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
            chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
            chunk = chunks[0]['RSSI Value'].tolist()
            plot_dict[f'{R}R {Q}Q'] = chunk
            # plot_dict['RSSI MEAN'] = raw_means[0][:len(chunk)]
            # plot_dict['LIMIT UP'] = bounds_up[0][:len(chunk)]
            # plot_dict['LIMIT DOWN'] = bounds_down[0][:len(chunk)]
    else:
        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
        plot_dict['REFERENCE'] = chunks[0]
        kalman_filter_par = config.KALMAN_1
        kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)
        chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
        plot_dict['SELECTED'] = chunks[0]

    df = pd.DataFrame(plot_dict)

    # df.plot.line(subplots=True)
    df.plot.line(
        title="Extremes of kalman Filter",
        xlabel="Sample",
        ylabel="RSSI(db)"
    )

    plt.savefig("plots/Kalman_different_kalman_filter.png")


def plot_dataset_without_outliers():
    X_with_outlier, _ = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")
    X_wo_outlier_r, _ = dataset_generator.generate_dataset_without_outliers("BLE2605r", "2605r0")
    X_wo_outlier_k, _ = dataset_generator.generate_dataset_without_outliers("BLE2605r", "2605r0", where_to_calc_mean=1)

    l = [
        X_with_outlier[:, 0].tolist(),
        X_wo_outlier_r[:, 0].tolist(),
        X_wo_outlier_k[:, 0].tolist()
    ]

    data = utility.equalize_data_with_nan(l)
    plot_dict = {
        "X_with_outlier": data[0],
        "X_without_outlier_raw": data[1],
        "X_without_outlier_kalman": data[2]
    }

    plot_df = pd.DataFrame(plot_dict)

    plot_df.plot.line(subplots=True)
    plot_df.plot.line()


def get_raws_means_and_bounds_for_plot(index_cuts, raw_chunks):
    raw_means = []
    bounds_up = []
    bounds_down = []

    for raw_chunks_reader, index_cut_reader in zip(raw_chunks, index_cuts):
        raw_mean = []
        for raw_chunk, start, end in zip(raw_chunks_reader, index_cut_reader['start'], index_cut_reader['end']):
            raw_rssi_mean = np.mean(raw_chunk['RSSI Value'])
            raw_mean.extend([raw_rssi_mean] * (end - start + 1))
        raw_means.append(raw_mean)

    for reader in raw_means:
        bounds_up.append([elem + 2. for elem in reader])
        bounds_down.append([elem - 2. for elem in reader])

    return raw_means, bounds_up, bounds_down


def plot_y_dataset(name_files_reader=config.NAME_FILES, name_files_cam=config.CAM_FILES):
    ax = plt.axes(title="Arrangement of points in the dataset")
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(name_files_reader)))

    i = 0
    for name_file_reader, name_file_cam in zip(name_files_reader, name_files_cam):
        _, y = dataset_generator.generate_dataset_base(name_file_reader, name_file_cam)
        p_x, p_y = utility.pol2cart(y[:, 0], y[:, 1])
        y = np.column_stack([p_x, p_y])
        df = pd.DataFrame(y, columns=["x(m)", "y(m)"])
        df.plot.scatter(x='x(m)', y="y(m)", ax=ax, color=colors[i], label=f'run{i}')
        i += 1

    plt.show()

    plt.savefig('plots/dataset_y.png')


def plot_good_points():
    df = pd.read_excel("kpc/kpc-good_points3.xlsx")
    df.plot.hexbin(x='R', y='Q', C='Nearest Neighbors D',
                   reduce_C_function=np.min,
                   gridsize=25,
                   cmap="viridis")
    df.plot.hexbin(x='R', y='Q', C='Nearest Neighbors D',
                   reduce_C_function=np.max,
                   gridsize=25,
                   cmap="viridis")
    df.plot.hexbin(x='R', y='Q', C='Nearest Neighbors D',
                   reduce_C_function=np.mean,
                   gridsize=25,
                   cmap="viridis")


if __name__ == "__main__":
    visualize = 7

    if visualize == 0:
        plot_kalman_rssi()

    if visualize == 1:
        plot_reader_rssi_stats()

    if visualize == 2:
        plot_raw_and_kalman_rssi()

    if visualize == 3:
        different_kalman_filter()

    if visualize == 4:
        specific_kalman_filter()

    if visualize == 5:
        plot_dataset_without_outliers()

    if visualize == 6:
        plot_y_dataset()

    plot_good_points()
