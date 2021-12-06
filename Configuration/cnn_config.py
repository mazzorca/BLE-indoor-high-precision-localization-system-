"""
Configuration file of the CNN
"""
import torchvision
from cnns_models.rfid_cnn import RFIDcnn
from cnns_models.wifi_cnn import WiFicnn
from cnns_models.ble_cnn import BLEcnn

NUMBER_ARGMAX_SQUARE = 1  # default number of square to use in the inference, if the metric is square
NUMBER_ARGMAX_EUCLIDEAN = 6  # default number of square to use in the inference, if the metric is euclidean

image_w = 15  # default width
image_h = 15  # default height

# transformation for the most popular cnns
transform_base_rgb = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((144, 144)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# transformation for the papers cnns
transform_base_grey = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(144),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# A dict with all the cnns taken into account
MODELS = {
    'alexnet': {
        'model': torchvision.models.alexnet(num_classes=18),
        'transform': transform_base_rgb
    },
    'resnet50': {
        'model': torchvision.models.resnet50(num_classes=18),
        'transform': transform_base_rgb
    },
    'squeezenet': {
        'model': torchvision.models.squeezenet1_0(num_classes=18),
        'transform': transform_base_rgb
    },
    'mobilenet_v3_small': {
        'model': torchvision.models.mobilenet_v3_small(num_classes=18),
        'transform': transform_base_rgb
    },
    'rfid': {
        'model': RFIDcnn(),
        'transform': torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize([0.5], [0.5])
            ])
         },
    'wifi': {
        'model': WiFicnn(),
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.Resize((16, 18)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
    },
    'ble': {
        'model': BLEcnn(),
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.Resize((24, 24)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
    }
}

active_moodels = {
    'alexnet': False,
    'resnet50': False,
    'squeezenet': False,
    'mobilenet_v3_small': False,
    'rfid': False,
    'wifi': False,
    'ble': True
}
