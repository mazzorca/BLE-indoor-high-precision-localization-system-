"""
File that contains the functions to transform the data in dataset for the regressors, cnn and rnn
"""
import math
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

from Configuration import dataset_config


def create_kalman_filter(kalman_filter_par):
    """
    create the kalman filter object
    :param kalman_filter_par: parameters to be used
    :return: the kalman filter object
    """
    kalman_filter = KalmanFilter(dim_x=1, dim_z=1)
    kalman_filter.F = np.array([[kalman_filter_par['A']]])
    kalman_filter.H = np.array([[kalman_filter_par['C']]])
    kalman_filter.R = np.array([[kalman_filter_par['R']]])
    kalman_filter.Q = np.array([[kalman_filter_par['Q']]])

    return kalman_filter


def get_index_taglio(tele):
    """
    Get the index on which the tag has been moved from one position to another from the camera data
    :param tele: array of data of the camera
    :return: array of index
    """
    indextaglio = [100]
    for i in range(len(tele[2]) - 1):  # len(tele[2]) = lunghezza valori nella x
        if (abs(tele[2][i] - tele[2][i + 1]) > 0.05) | (abs(tele[3][i] - tele[3][i + 1]) > 0.05):
            indextaglio.append(i)

    indextaglio.append(len(tele[2]) - 1)

    return indextaglio


def get_index_taglio_reader(time):
    """
    Get the index on which the tag has been moved from one position to another from the readers data
    :param time: array of timestamp on which the reader receives the beacon
    :return: array of index
    """
    indextaglio_reader = []

    for j in range(5):
        indextaglio_reader.append([0])
        for i in range(len(time[j]) - 1):
            if abs(time[j][i] - time[j][i + 1]) > 5:  # 5 secondi
                indextaglio_reader[j].append(i)
        indextaglio_reader[j].append(len(time[j]) - 1)

    # for j in range(2):
    #     for i in range(len(indextaglio_reader)):
    #         indextaglio_reader.append(indextaglio_reader[i])

    return indextaglio_reader


def fixReader(dati, time, tele):
    """
    Function for cutting data from the 5 readers, aligning the data collected by the cameras and the data collected by the readers over time over time
    :param dati: data from the readers
    :param time: time from the readers
    :param tele: data from the cameras
    :return: extended reader data, data from the cameras and the new index of change position
    """
    newData = []
    newTime = []

    indextaglio = get_index_taglio(tele)
    indextaglio_reader = get_index_taglio_reader(time)

    num_tagli = 0
    for i in range(len(indextaglio_reader[0]) - 1):
        if indextaglio_reader[0][i + 1] - indextaglio_reader[0][i] >= 1:
            num_tagli += 1

    print(num_tagli)

    # fig, ax = plt.subplots()
    # debug_index_taglio_dict = {}
    # for j in range(0):
    #     debug_index_taglio = []
    #     for i in range(len(time[j]) - 1):
    #         elem = time[j][i + 1] - time[j][i]
    #         debug_index_taglio.append(elem)
    #     x_axis = list(range(len(debug_index_taglio)))
    #     ax.plot(x_axis, debug_index_taglio)

    # df = pd.DataFrame(debug_index_taglio_dict)
    # df.plot.line()

    if len(indextaglio) < len(indextaglio_reader[0]):
        for index in indextaglio_reader:
            index.remove(index[0])

    for i in range(len(dati)):
        newData.append([])
        newTime.append([])
        for j in range(len(indextaglio_reader[i]) - 1):
            tot_data_inserted = 0
            dim_data_reader = indextaglio_reader[i][j + 1] - indextaglio_reader[i][j]  # dimensione dei dati raccolti

            # nel punto j-esimo dal BLE
            dim_data_tele = indextaglio[j + 1] - indextaglio[j]  # dimensione dei dati raccolti nel punto j-esimo
            # dalle tele
            fattore = math.trunc(dim_data_tele / dim_data_reader)

            for t in range(dim_data_reader + 1):
                for x in range(fattore + 1):
                    if tot_data_inserted < dim_data_tele:
                        newData[i].append(dati[i][indextaglio_reader[i][j] + t + 1])
                        newTime[i].append(time[i % 5][indextaglio_reader[i][j] + t + 1])
                        tot_data_inserted += 1

    return newData, newTime, indextaglio


def transform_in_dataframe(data, index):
    """
    transform the dataset in a list o dataframe in witch each dataframe represent a position
    :param data: data from the readers
    :param index: index of change position
    :return: list of dataframe with five columns corresponding to the RSSI value of the readers
    """
    reader_position_list = {}
    for index_row, row in index[0].iterrows():
        reader_position_list[f'pos{index_row}'] = pd.DataFrame()

    for i in range(len(data)):
        for index_row, row in index[i].iterrows():
            position_reader_data = data[i][row['start']: row['end']]
            position_reader_df = pd.DataFrame({f'reader{i}': position_reader_data})

            reader_position_list[f'pos{index_row}'] = pd.concat(
                [
                    reader_position_list[f'pos{index_row}'],
                    position_reader_df
                ], axis=1)

    for position in reader_position_list.keys():
        df = reader_position_list[position]
        first_cut_number = int(df.shape[0] / dataset_config.INITIAL_CUT)
        df = df.iloc[first_cut_number:]
        final_cut_number = int(df.shape[0] / dataset_config.FINAL_CUT)
        reader_position_list[position] = df.iloc[:-final_cut_number]

    for position in reader_position_list.keys():
        df = reader_position_list[position]
        reader_position_list[position] = df.loc[(df.shift() != df).any(axis=1)]

    return reader_position_list


def cutReader(dati, tele, ind):
    """
    remove the first one third and last one third of each position and remove the duplicates
    :param dati: data from the readers
    :param tele: data from the comeras
    :param ind: index of change position
    :return: cutted data from reader and cameras and new list of indexes of change position
    """
    newData = [[] for _ in range(len(dati))]
    newTele = [[] for _ in range(2)]
    new_data = [0 for _ in range(len(dati))]
    old_data = [0 for _ in range(len(dati))]

    new_ind = [0]
    for j in range(len(ind) - 1):
        dim = ind[j + 1] - ind[j]
        offset = math.trunc(dim / 3)
        # print(offset)
        lun = offset
        for t in range(lun):
            for i in range(len(dati)):
                if ind[j] + offset + t < len(dati[i]):
                    new_data[i] = dati[i][ind[j] + offset + t]

            # np.isnan, is for mean and std dataset
            if not np.array_equal(new_data, old_data) and not np.isnan(np.min(new_data)):
                for i in range(len(dati)):
                    newData[i].append(dati[i][ind[j] + offset + t])
                for i in range(len(new_data)):
                    old_data[i] = new_data[i]
                newTele[0].append(tele[2][ind[j] + offset + t])
                newTele[1].append(tele[3][ind[j] + offset + t])
        new_ind.append(len(newData[0]))

    # for i in range (5):
    # print(len(newData[i]))
    return newData, newTele, new_ind


def apply_kalman_filter(no_kalman_data, kalman_filter_par):
    """
    Apply  the kalman filter on the raw data
    :param no_kalman_data: raw data from the readers
    :param kalman_filter_par: kalman filter parameters to be used
    :return: Data from readers filtered with the kalman filter
    """
    kalman_data = []

    kalman_filter = create_kalman_filter(kalman_filter_par)

    for i, raw_data_reader in enumerate(no_kalman_data):
        # reset the filter
        kalman_filter.x = np.array([kalman_filter_par['x']])
        kalman_filter.P = np.array([[kalman_filter_par['P']]])

        kalman_data.append([])
        for raw_value in raw_data_reader:
            kalman_filter.predict()
            kalman_filter.update(raw_value)
            kalman_data[i].append(kalman_filter.x[0])

    return kalman_data


def positivize_rssi(data_readers, min_RSSI):
    """
    For RNN process the RSSI by adding the min value to get all positive value
    :param data_readers: data from readers
    :param min_RSSI: min value to add
    :return: the data from reader in positive value
    """
    normalized_RSSI = [[] for _ in data_readers]

    for r, data_reader in enumerate(data_readers):
        for RSSI_value in data_reader:
            normalized_RSSI[r].append(RSSI_value + abs(min_RSSI))

    return normalized_RSSI
