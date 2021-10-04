import pandas as pd
import numpy as np

import RSSI_image_converter
import data_converter
import utility
import data_extractor
import config
from Configuration import dataset_config

dataset_tests = {
    "dati3105run0r dati3105run1r dati3105run2r": ["x_test", "y_test"],
    "dati3105run0r": ["x_test0", "y_test0"],
    "dati3105run1r": ["x_test1", "y_test1"],
    "dati3105run2r": ["x_test2", "y_test2"]
}


def get_processed_data_from_a_kalman_data(kalman_data, time, name_file_cam):
    dati_cam = utility.convertEMT(name_file_cam)
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(kalman_data, time, dati_cam)
    final_data_reader, final_data_cam, _ = data_converter.cutReader(dati_reader_fixed, dati_cam, index_cut)

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
    kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)

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
    kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)

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


def generate_dataset(name_file_reader, name_file_cam, type_of_dataset):
    X_a = []
    y_a = []
    for name_file, cam_file in zip(name_file_reader, name_file_cam):
        x_r, y_r = type_of_dataset(name_file, cam_file)
        print(x_r.shape)
        X_a.append(x_r)
        y_a.append(y_r)

    X, y = concatenate_dataset(X_a, y_a)

    return X, y


def save_dataset_numpy_file(RSSI_file, RSSI, position_file, position):
    with open(f'datasets/{RSSI_file}.npy', 'wb') as f:
        np.save(f, RSSI)

    with open(f'datasets/{position_file}.npy', 'wb') as f:
        np.save(f, position)


def load_dataset_numpy_file(RSSI_file, position_file):
    with open(f'datasets/{RSSI_file}.npy', 'rb') as f:
        RSSI = np.load(f)

    with open(f'datasets/{position_file}.npy', 'rb') as f:
        position = np.load(f)

    return [RSSI, position]


def create_image_dataset(name_file_reader, name_file_cam, w, h, stride, kalman_filter=None):
    """

    :param name_file_reader:
    :param name_file_cam:
    :param w:
    :param h:
    :param stride:
    :param kalman_filter:
        - Not use Kalman
        - 1: kalman base
    :return:
    """
    dati_cam = utility.convertEMT(name_file_cam)
    data, time = data_extractor.get_raw_rssi_csv(name_file_reader)

    min = dataset_config.NORM_MIN_NK
    max = dataset_config.NORM_MAX_NK
    if kalman_filter:
        print("Using Kalman")
        if kalman_filter == 1:
            kalman_filter = config.KALMAN_BASE
        data = data_converter.apply_kalman_filter(data, kalman_filter)
        min = dataset_config.NORM_MIN_K
        max = dataset_config.NORM_MAX_K

    normalized_data = RSSI_image_converter.normalize_rssi(data, min, max)
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(normalized_data, time, dati_cam)
    index = utility.get_index_start_and_end_position(time_fixed)
    list_of_position = data_converter.transform_in_dataframe(dati_reader_fixed, index)
    dati_cam = [dati_cam[2], dati_cam[3]]
    labels = RSSI_image_converter.get_label(dati_cam, index_cut)

    final_dir = f'{w}x{h}-{stride}/{name_file_reader}'
    RSSI_image_converter.translate_RSSI_to_image_greyscale(list_of_position, labels, final_dir, w, h, stride)


def create_matrix_dataset(name_file_reader, name_file_cam, kalman_filter=None):
    dati_cam = utility.convertEMT(name_file_cam)
    data, time = data_extractor.get_raw_rssi_csv(name_file_reader)

    min = dataset_config.NORM_MIN_NK
    if kalman_filter:
        print("Using Kalman")
        if kalman_filter == 1:
            kalman_filter = config.KALMAN_BASE
        data = data_converter.apply_kalman_filter(data, kalman_filter)
        min = dataset_config.NORM_MIN_K

    normalized_data = data_converter.positivize_rssi(data, min)
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(normalized_data, time, dati_cam)
    index = utility.get_index_start_and_end_position(time_fixed)
    list_of_position = data_converter.transform_in_dataframe(dati_reader_fixed, index)
    dati_cam = [dati_cam[2], dati_cam[3]]
    labels = RSSI_image_converter.get_label(dati_cam, index_cut)

    windows_size = dataset_config.WINDOW_SIZE_RNN
    stride = 10
    PATH = f"datasets/rnn_dataset/{name_file_reader}/"
    csv_file = f"{PATH}matrix.csv"
    utility.check_and_if_not_exists_create_folder(csv_file)
    utility.append_to_csv(csv_file, [["RSSI", "optimal_x", "optimal_y"]])
    for position_str, label in zip(list_of_position.keys(), labels):
        folder_name = f"{PATH}{position_str}/"
        utility.check_and_if_not_exists_create_folder(folder_name)
        position_df = list_of_position[position_str]

        csv_list = []
        for i in range(0, position_df.shape[0] - windows_size, stride):
            sequence_df = position_df.iloc[i:i + windows_size, :]

            max_df = pd.DataFrame()
            for j in range(0, sequence_df.shape[0], stride):
                second_df = sequence_df.iloc[j: j+stride, :]
                max_df = pd.concat([max_df, pd.DataFrame(second_df.max()).transpose()], axis=0)

            file_name = f"{folder_name}/matrix{i}.npy"
            csv_file_name = f"{position_str}/matrix{i}.npy"

            csv_list.append([csv_file_name, label[1], label[2]])

            with open(file_name, 'wb') as f:
                RSSI = max_df.to_numpy()
                np.save(f, RSSI)

        utility.append_to_csv(csv_file, csv_list)


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
