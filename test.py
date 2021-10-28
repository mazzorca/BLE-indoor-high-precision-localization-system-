import torchvision
from torch.utils.data import DataLoader

import config
import data_converter
from Configuration import cnn_config
from RSSI_images_Dataset import RSSIImagesDataset
from dataset_generator import create_image_dataset, create_matrix_dataset
from cnns_models.ble_cnn import BLEcnn
from torchinfo import summary

from rnn_dataset import RnnDataset
from rnns_models import ble
from torch.utils.tensorboard import SummaryWriter

import utility
import data_extractor
import RSSI_image_converter

if __name__ == "__main__":
    # name_files_reader = ["dati3105run2r"]
    # name_files_cam = ["Cal3105run2"]
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

    # params = [
    #     #     [15, 15, 10],
    #     [20, 20, 10]
    #     #     [25, 25, 10],
    #     #     [5, 60, 10]
    # ]
    # # #
    #
    # for param in params:
    #     print("w:", param[0], "h:", param[1], "stride:", param[2])
    #     for name_file_reader, name_file_cam in zip(config.NAME_FILES, config.CAM_FILES):
    #         create_image_dataset(name_file_reader, name_file_cam, param[0], param[1], param[2], 1)

    # for name_file_reader, name_file_cam in zip(config.NAME_FILES, config.CAM_FILES):

    # for name_file_reader, name_file_cam in zip(config.NAME_FILES, config.CAM_FILES):
    #     print("generating  dataset", name_file_reader)
    #     create_matrix_dataset(name_file_reader, name_file_cam, 1)

    writer = SummaryWriter('runs/')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_set = RnnDataset(csv_file=f"datasets/rnn_dataset/BLE2605r/matrix.csv",
                           root_dir=f"datasets/rnn_dataset/BLE2605r",
                           transform=transform)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1)

    dataiter = iter(train_loader)
    matrix, labels = dataiter.next()

    net = ble.BLErnn()
    # print(summary(net, input_size=(1, 24, 24)))

    # net = BLEcnn()

    writer.add_graph(net.float(), matrix.float())
    writer.close()

    # raws_data, time = data_extractor.get_raw_rssi_csv("dati3105run2r")
    # kalman_filter_par = config.KALMAN_BASE
    # kalman_data = data_converter.apply_kalman_filter(raws_data, kalman_filter_par)
    # max, min = RSSI_image_converter.get_max_and_min_of_data(kalman_data)
    # print("max", max, "min", min)
