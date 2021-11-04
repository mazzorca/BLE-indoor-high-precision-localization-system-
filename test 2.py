import numpy as np
from PIL import Image
import torch

import config
import data_converter
from Configuration import cnn_config
from dataset_generator import create_image_dataset, create_matrix_dataset
from cnns_models.ble_cnn import BLEcnn
from torchinfo import summary
from rnns_models import ble

import utility
import data_extractor
import RSSI_image_converter
from cnn_testing import load_model

if __name__ == "__main__":
    model_name = "ble"
    kalman = "kalman"
    transform = cnn_config.MODELS[model_name]["transform"]
    model_cnn = cnn_config.MODELS[model_name]["model"]

    params = {
        "wxh-stride": "20x20-10",
        "epoch": 20,
        "batch_size": 32,
        "lr": 0.01
    }

    model_name = f"{model_name}_{kalman}"
    parameters_saved = f"{model_name}/{int(params['epoch'])}-{params['lr']}-{int(params['batch_size'])}-{params['wxh-stride']}"
    model_cnn = load_model(model_cnn, parameters_saved)
    model_cnn.eval()

    image_np = np.random.rand(20, 20)*255
    image_np = image_np.astype(np.uint8)
    image_np = np.array([image_np, image_np, image_np])
    print(image_np)
    img = Image.fromarray(image_np, 'RGB')
    # img = img.convert('RGB')

    tensor_img = transform(img)
    tensor_img = tensor_img.view([1, 1, 24, 24])

    tensor_np = tensor_img.numpy()

    with torch.no_grad():
        pred = model_cnn(tensor_img)
        probability = torch.nn.functional.softmax(pred, dim=1)

        probability_np = probability.cpu().numpy()[0]
        indexs = probability_np.argsort()[-18:]

        normalized_sum = np.sum(probability_np[indexs])

        x = 0
        y = 0
        for index in indexs:
            normalized_probability = probability_np[index] / normalized_sum

            contribution_x = config.SQUARES[index].centroid.x * normalized_probability
            contribution_y = config.SQUARES[index].centroid.y * normalized_probability
            x += contribution_x
            y += contribution_y

        print("x:", x, "y", y)
