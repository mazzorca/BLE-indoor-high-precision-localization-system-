import pandas as pd
import numpy as np

import utility
import data_extractor
import config


def get_processed_data_from_a_kalman_data(kalman_data, time, name_file_cam):
    dati_cam = utility.convertEMT(name_file_cam)
    dati_reader_fixed, time_fixed, index_cut = utility.fixReader(kalman_data, time, dati_cam)
    final_data_reader, final_data_cam = utility.cutReader(dati_reader_fixed, dati_cam, index_cut)

    return final_data_reader, final_data_cam


def generate_dataset_from_final_data(data_reader, data_cam):
    X = np.array(data_reader)
    X = np.transpose(X)

    cam_cut_np = np.array(data_cam)
    cam_cut_np = np.transpose(cam_cut_np)

    ry, py = utility.cart2pol(cam_cut_np[:, 0], cam_cut_np[:, 1])
    y = np.column_stack([ry, py])

    return X, y


def generate_dataset_base(name_file_reader, name_file_cam):
    kalman_filter_par = config.KALMAN_BASE
    kalman_data, kalman_time = utility.extract_and_apply_kalman_csv(name_file_reader, kalman_filter_par)
    X_rssi, y_rssi = get_processed_data_from_a_kalman_data(kalman_data, kalman_time, name_file_cam)
    X, y = generate_dataset_from_final_data(X_rssi, y_rssi)

    return X, y


def generate_dataset_from_list_of_files(name_file_readers, name_file_cams):
    X = np.array([[] for _ in range(5)]).transpose()
    y = np.array([[] for _ in range(2)]).transpose()
    for name_file, cam_file in zip(name_file_readers, name_file_cams):
        print(name_file)
        X_run, y_run = generate_dataset_base(name_file, cam_file)
        X = np.concatenate((X, X_run))
        y = np.concatenate((y, y_run))

    return X, y


def generate_dataset_base_all():
    name_files = ["BLE2605r", "dati3105run0r", "dati3105run1r", "dati3105run2r"]
    cam_files = ["2605r0", "Cal3105run0", "Cal3105run1", "Cal3105run2"]

    X, y = generate_dataset_from_list_of_files(name_files, cam_files)

    return X, y


def concatenate_dataset(datasets, cams):
    X = np.array([[] for _ in range(datasets[0].shape[1])]).transpose()
    y = np.array([[] for _ in range(2)]).transpose()
    for dataset, cam in zip(datasets, cams):
        X = np.concatenate((X, dataset))
        y = np.concatenate((y, cam))

    return X, y


def generate_dataset_without_outliers(name_file_reader, name_file_cam, where_to_calc_mean=0):
    """

    :param name_file_reader:
    :param name_file_cam:
    :param where_to_calc_mean:  0: on raw rssi
                                1: on kalman rssi
    :return:
    """
    raws_data, raws_time = data_extractor.get_raw_rssi_csv(name_file_reader)

    kalman_filter_par = config.KALMAN_BASE
    kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)

    index_cut = utility.get_index_start_and_end_position(raws_time)
    kalman_chunks = utility.get_chunk(kalman_data, index_cut)
    raw_chunks = utility.get_chunk(raws_data, index_cut)

    chunks = raw_chunks
    if where_to_calc_mean == 1:
        chunks = kalman_chunks
    _, bounds_ups, bounds_downs = utility.get_means_and_bounds(chunks)
    kalman_without_outliers = utility.remove_outliers(kalman_chunks, bounds_ups, bounds_downs)

    f_data_reader, f_data_cam = get_processed_data_from_a_kalman_data(kalman_without_outliers, raws_time, name_file_cam)
    X, y = generate_dataset_from_final_data(f_data_reader, f_data_cam)

    return X, y


def generate_dataset_with_mean_and_std(name_file_reader, name_file_cam, kalman_filter_par=None):
    raws_data, raws_time = data_extractor.get_raw_rssi_csv(name_file_reader)
    index_cut_reader = utility.get_index_start_and_end_position(raws_time)
    raw_chunks = utility.get_chunk(raws_data, index_cut_reader)

    if kalman_filter_par is None:
        kalman_filter_par = config.KALMAN_BASE
    kalman_data = utility.apply_kalman_filter(raws_data, kalman_filter_par)

    kalman_data_w_mean_std = utility.add_mean_and_std(kalman_data, raw_chunks)
    f_data_reader, f_data_cam = get_processed_data_from_a_kalman_data(kalman_data_w_mean_std, raws_time, name_file_cam)
    X, y = generate_dataset_from_final_data(f_data_reader, f_data_cam)

    return X, y


def generate_dataset_with_mean_and_std_all():
    X_a = []
    y_a = []
    for name_file, cam_file in zip(config.NAME_FILES, config.CAM_FILES):
        X_r, y_r = generate_dataset_with_mean_and_std(name_file, cam_file)
        X_a.append(X_r)
        y_a.append(y_r)

    X, y = concatenate_dataset(X_a, y_a)

    return X, y


if __name__ == "__main__":
    datiCSV1, datiEMT1 = utility.takeData("dati3105run0r", "Cal3105run0")
    utility.printDati(datiCSV1, datiEMT1)
    print(len(datiCSV1[0]))
    print(len(datiEMT1[0]))

    datiCSV2, datiEMT2 = utility.takeData("dati3105run1r", "Cal3105run1")
    utility.printDati(datiCSV2, datiEMT2)
    print(len(datiCSV2[0]))
    print(len(datiEMT2[0]))

    datiCSV3, datiEMT3 = utility.takeData("dati3105run2r", "Cal3105run2")
    utility.printDati(datiCSV3, datiEMT3)
    print(len(datiCSV3[0]))
    print(len(datiEMT3[0]))

    datiCSV0, datiEMT0 = utility.takeData("BLE2605r", "2605r0")
    utility.printDati(datiCSV0, datiEMT0)
    print(len(datiCSV0[0]))
    print(len(datiEMT0[0]))

    name = "Train"
    utility.saveDataArff(datiCSV0, datiEMT0, name)
    print("fine savedataarff 0")

    name = "Test1"
    utility.saveDataArff(datiCSV1, datiEMT1, name)
    print("fine savedataarff 1")

    name = "Test2"
    utility.saveDataArff(datiCSV2, datiEMT2, name)
    print("fine savedataarff 2")

    name = "Test3"
    utility.saveDataArff(datiCSV3, datiEMT3, name)
    print("fine savedataarff 3")

    X = [[], [], [], [], []]
    Y = [[], []]
    for i, x in enumerate(X):
        x.extend(datiCSV0[i])
        x.extend(datiCSV3[i])
        x.extend(datiCSV2[i])
        x.extend(datiCSV1[i])
        print(len(x))

    for i, y in enumerate(Y):
        y.extend(datiEMT0[i])
        y.extend(datiEMT3[i])
        y.extend(datiEMT2[i])
        y.extend(datiEMT1[i])
        print(len(y))

    name = "Train0"
    utility.saveDataArff(X, Y, name)
    print("fine savedataarff 00")
