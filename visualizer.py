"""
This file contains the functions to visualize data
"""
import matplotlib.pyplot as plt

import config

import math
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import data_converter
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
    kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
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
    kalman_data1 = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)

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


def plot_kalman_and_extended_kalman_rssi():
    kalman_filter_par = config.KALMAN_BASE
    kalman_data, kalman_time = utility.extract_and_apply_kalman_csv("BLE2605r", kalman_filter_par)
    index_kalman = utility.get_index_start_and_end_position(kalman_time)
    kalman_chunks = utility.get_chunk(kalman_data, index_kalman)

    dati_cam = utility.convertEMT("2605r0")
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(kalman_data, kalman_time, dati_cam)
    index_extended = utility.get_index_start_and_end_position(time_fixed)
    extended_chunks = utility.get_chunk(dati_reader_fixed, index_extended)

    kalman_df = pd.DataFrame({"KALMAN_RSSI_0": kalman_chunks[0][0]['RSSI Value']})
    extended_df = pd.DataFrame({"Extended_KALMAN_RSSI_0": extended_chunks[0][0]['RSSI Value']})

    # df.plot.line(subplots=True)
    kalman_df.plot.line(
        title="filtered RSSI",
        xlabel="Sample",
        ylabel="RSSI(db)"
    )

    plt.savefig("plots/kalman_rssi.png")

    extended_df.plot.line(
        title="extended RSSI",
        xlabel="Sample",
        ylabel="RSSI(db)"
    )

    plt.savefig("plots/extended_rssi.png")


def plot_extended_kalman_and_cutted_rssi():
    kalman_filter_par = config.KALMAN_BASE
    kalman_data, kalman_time = utility.extract_and_apply_kalman_csv("BLE2605r", kalman_filter_par)

    dati_cam = utility.convertEMT("2605r0")
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(kalman_data, kalman_time, dati_cam)
    index_extended = utility.get_index_start_and_end_position(time_fixed)
    extended_chunks = utility.get_chunk(dati_reader_fixed, index_extended)

    final_data_reader, final_data_cam, _ = data_converter.cutReader(dati_reader_fixed, dati_cam, index_cut)

    extended_df = pd.DataFrame({"Extended_KALMAN_RSSI_0": extended_chunks[0][0]['RSSI Value']})
    cutted_df = pd.DataFrame({"Cutted_KALMAN_RSSI_0": final_data_reader[0][0:470]})

    extended_df.plot.line(
        title="extended RSSI",
        xlabel="Sample",
        ylabel="RSSI(db)"
    )

    plt.savefig("plots/extended_rssi.png")

    cutted_df.plot.line(
        title="Cutted RSSI",
        xlabel="Sample",
        ylabel="RSSI(db)"
    )

    plt.savefig("plots/cutted_rssi.png")


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
    kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
    plot_dict['REFERENCE'] = kalman_data[0]

    R = 0.025
    # Q = 0.00001
    for Q in np.linspace(.0001, .000001, 11):
        utility.change_Q_R(kalman_filter_par, R, Q)
        kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
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
    kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
    plot_dict['REFERENCE'] = kalman_data[0]

    if kalman_filters is not None:
        for index, row in kalman_filters.iterrows():
            R = row['R']
            kalman_filter_par['R'] = R
            Q = row['Q']
            kalman_filter_par['Q'] = Q
            kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
            plot_dict[f'{index} - {R}R {Q}Q'] = kalman_data[0]
    else:
        kalman_filter_par = config.KALMAN_1
        kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
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
            kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
            chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
            chunk = chunks[0]['RSSI Value'].tolist()
            plot_dict[f'{R}R {Q}Q'] = chunk
            # plot_dict['RSSI MEAN'] = raw_means[0][:len(chunk)]
            # plot_dict['LIMIT UP'] = bounds_up[0][:len(chunk)]
            # plot_dict['LIMIT DOWN'] = bounds_down[0][:len(chunk)]
    else:
        kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
        chunks = utility.get_chunk(kalman_data, index_cut, chunk_num=selected_cut)
        plot_dict['REFERENCE'] = chunks[0]
        kalman_filter_par = config.KALMAN_1
        kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
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


def plot_good_points_sparse(name_file="kpc/kpc-good_points_high_range.xlsx", regressor='Nearest Neighbors D'):
    df = pd.read_excel(name_file)

    plot_dict = {
        'R': [],
        'Q': [],
        regressor: []
    }
    for index, row in df.iterrows():
        # 0.000000000000001
        plot_dict['R'].append(math.floor(math.log10(row['R'])))
        plot_dict['Q'].append(math.floor(math.log10(row['Q'])))
        plot_dict[regressor].append(row[regressor])

    plot_df = pd.DataFrame(plot_dict)

    plot_df.plot.hexbin(x='R', y='Q', C=regressor,
                        reduce_C_function=np.mean,
                        gridsize=8,
                        cmap="viridis",
                        title=regressor)

    plt.savefig(f'plots/hexbin_{regressor}.png')


def plot_good_points_line(what_to_plot, name_file="kpc/kpc-good_points.xlsx", regressor='Nearest Neighbors D'):
    df = pd.read_excel(name_file)

    plot_dict = {}
    R_list = list(set(df['R'].tolist()))
    Q_list = list(set(df['Q'].tolist()))
    R_list.sort()
    Q_list.sort()
    plot_dict['x'] = np.array(R_list) if what_to_plot == 'Q' else np.array(Q_list)

    for index, row in df.iterrows():
        key = row[what_to_plot]

        good_points = row[regressor]
        utility.create_or_insert_in_list(plot_dict, f'{what_to_plot}-{key}', good_points)

    plot_df = pd.DataFrame(plot_dict)

    x_label = 'R' if what_to_plot == 'Q' else 'Q'
    plot_df.plot.line(x='x',
                      title="Varying of predicted points",
                      xlabel=x_label,
                      ylabel="predicted Points")
    plt.savefig(f'plots/predicted_points{what_to_plot}.png')


def plot_table_plygons():
    squares = config.SQUARES

    ax = plt.axes(title="Table")

    for square in squares:
        x, y = square.exterior.xy
        ax.plot(x, y)

    plt.show()


def plot_3d_setting_time_and_predicted_point(predicted_name_files, settling_name_file,
                                             regressor_name="Nearest Neighbors D",
                                             settling_name="SETTLING_SAMPLE_AVG"):
    predicted_df = pd.read_excel(predicted_name_files)
    settling_df = pd.read_excel(settling_name_file)

    ax = plt.figure().add_subplot(projection='3d')

    R_log = []
    Rs = []
    Q_log = []
    Qs = []
    for index, row in predicted_df.iterrows():
        R = row['R']
        R_log.append(math.log10(R))
        Rs.append(R)

        Q = row['Q']
        Q_log.append(math.log10(Q))
        Qs.append(Q)

    R_log = list(set(R_log))
    Q_log = list(set(Q_log))
    Rs = list(set(Rs))
    Qs = list(set(Qs))
    R_log.sort()
    Q_log.sort()
    Rs.sort()
    Qs.sort()

    RX = np.array(R_log)
    xlen = len(RX)
    QY = np.array(Q_log)
    ylen = len(QY)
    RX, QY = np.meshgrid(RX, QY)

    Z = np.array(predicted_df[regressor_name].tolist())
    Z = Z.reshape(33, 33)

    # Create an empty array of strings with the same shape as the meshgrid, and
    # populate it with two colors in a checkerboard pattern.
    max_settling = settling_df[settling_name].max()
    min_settling = settling_df[settling_name].min()
    color_set = plt.get_cmap("viridis")
    colors = np.empty(RX.shape, dtype=object)
    normalized_settlings = []
    for y in range(ylen):
        for x in range(xlen):
            row = settling_df.loc[(settling_df['R'] == Rs[x]) & (settling_df['Q'] == Qs[y])]
            normalized_settling = row[settling_name].iloc[0] / max_settling
            normalized_settlings.append(normalized_settling)
            colors[y, x] = color_set(normalized_settling)

    # Plot the surface with face colors taken from the array we made.
    surf = ax.plot_surface(RX, QY, Z, facecolors=colors, linewidth=0)

    plt.savefig("plots/3dPlot.png")
    ax = plt.figure()
    normalized_settlings = np.array(normalized_settlings)
    normalized_settlings = np.multiply(normalized_settlings, max_settling)
    normalized_settlings = normalized_settlings.reshape(33, 33)
    plt.imshow(normalized_settlings, origin="lower", cmap='viridis', interpolation='nearest')
    plt.colorbar()

    plt.savefig("plots/ColorBar.png")

    plt.show()
