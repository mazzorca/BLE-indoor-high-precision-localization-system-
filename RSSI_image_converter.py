import numpy as np
from PIL import Image
import numpy as np
import csv

import utility
import data_converter
import config
from Configuration import dataset_config


def get_max_and_min_of_data(data):
    max_elem = -100
    min_elem = 0

    for d in data:
        temp_max = max(d)
        if max_elem < temp_max:
            max_elem = temp_max

        temp_min = min(d)
        if min_elem > temp_min:
            min_elem = temp_min

    return max_elem, min_elem


def normalize_rssi(data_readers, max_RSSI, min_RSSI):
    normalized_RSSI = [[] for _ in data_readers]

    dividend = max_RSSI - min_RSSI

    for r, data_reader in enumerate(data_readers):
        for RSSI_value in data_reader:
            normalized_RSSI[r].append((RSSI_value - min_RSSI) / dividend)

    return normalized_RSSI


def get_label(dati_cameras, index_cut):
    labels = []
    for i in range(len(index_cut) - 1):
        j = index_cut[i] + int((index_cut[i + 1] - index_cut[i]) / 2)

        px = dati_cameras[0][j]
        py = dati_cameras[1][j]

        square_x, square_y = utility.get_square_number(px, py, config.SQUARES)
        square_number = square_y * 6 + square_x

        labels.append([square_number, px, py])

    return labels


def translate_RSSI_to_image_greyscale(data, labels, name_experiment, w=15, h=15, stride=3):
    windows_size = int((w * h) / config.NUM_READERS)

    base_path = f"datasets/cnn_dataset/{name_experiment}/"
    utility.check_and_if_not_exists_create_folder(base_path)

    csv_file = f"{base_path}RSSI_images.csv"
    utility.append_to_csv(csv_file, [["RSSI", "Label", "optimal_x", "optimal_y"]])
    for position, label in zip(data.keys(), labels):
        position_df = data[position]

        directory_check = False
        IMAGE_PATH = f"{base_path}RSSI_images/square{label[0]}"
        path_for_csv = f"square{label[0]}"
        csv_list = []
        for i in range(0, position_df.shape[0] - windows_size, stride):
            image_df = position_df.iloc[i:i + windows_size, :]

            x = 0
            y = 0
            img_array = np.zeros((h, w), dtype=np.uint8)
            for index, row in image_df.iterrows():
                for r in range(config.NUM_READERS):
                    img_array[x, y] = row[f'reader{r}'] * 255
                    x += 1

                if x == h:
                    y += 1
                    x = 0

            img = Image.fromarray(img_array, 'L')
            filename = f'{IMAGE_PATH}/img{i}.jpg'
            csv_filename = f'{path_for_csv}/img{i}.jpg'

            if not directory_check:
                utility.check_and_if_not_exists_create_folder(filename)
                directory_check = True

            img.save(filename)
            csv_list.append([csv_filename, label[0], label[1], label[2]])

        utility.append_to_csv(csv_file, csv_list)


def RSSI_numpy_to_image_greyscale(image_np, w=15, h=15):
    x = 0
    y = 0
    img_array = np.zeros((h, w), dtype=np.uint8)
    for index, row in image_np:
        for r in range(config.NUM_READERS):
            img_array[x, y] = row[f'reader{r}'] * 255
            x += 1

        if x == h:
            y += 1
            x = 0

    img = Image.fromarray(img_array, 'L')

    return img
