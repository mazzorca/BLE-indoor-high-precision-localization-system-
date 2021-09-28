import config
import data_converter
from dataset_generator import create_image_dataset
from cnns_models.ble_cnn import BLEcnn
from torchinfo import summary

import utility
import data_extractor
import RSSI_image_converter

if __name__ == "__main__":
    # name_files_reader = ["BLE2605r"]
    # name_files_cam = ["2605r0"]

    # params = [
    #     [15, 15, 3],
    #     [15, 15, 10],
    #     [20, 20, 3],
    #     [20, 20, 10],
    #     [25, 25, 3],
    #     [25, 25, 10],
    #     [5, 45, 1],
    #     [5, 45, 3],
    #     [5, 45, 10],
    #     [5, 60, 3],
    #     [5, 60, 10]
    # ]

    params = [
        [15, 15, 10],
        [20, 20, 10],
        [25, 25, 10],
        [5, 60, 10]
    ]

    for param in params:
        print("w:", param[0], "h:", param[1], "stride:", param[2])
        for name_file_reader, name_file_cam in zip(config.NAME_FILES, config.CAM_FILES):
            create_image_dataset(name_file_reader, name_file_cam, param[0], param[1], param[2])

    # net = BLEcnn()
    # print(summary(net, input_size=(1, 1, 24, 24)))

    # raws_data, time = data_extractor.get_raw_rssi_csv("dati3105run2r")
    # kalman_filter_par = config.KALMAN_BASE
    # kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
    # max, min = RSSI_image_converter.get_max_and_min_of_data(kalman_data)
    # print("max", max, "min", min)


