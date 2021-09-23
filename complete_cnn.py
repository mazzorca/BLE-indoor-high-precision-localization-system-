from train_cnns import train_model
from cnn_testing import cnn_test, write_cnn_result, load_model
from Configuration import cnn_config

if __name__ == '__main__':
    model_name = "ble"
    model = cnn_config.MODELS[model_name]["model"]
    transform = cnn_config.MODELS[model_name]["transform"]

    wxh = "20x20-10"
    dataset = "BLE2605r"

    epochs = 20
    bs = 32
    lr = 0.01

    train_model(model, wxh, dataset, transform, epochs, lr, bs, model_name)

    testing_datasets = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    type_dist = 1

    for testing_dataset in testing_datasets:
        parameters_saved = f"{model_name}/{epochs}-{lr}-{bs}-{wxh}"
        model = load_model(model, parameters_saved)
        preds, ys = cnn_test(model, testing_dataset, transform, bs, type_dist)

        base_file_name = f'cnn_results/{model_name}/{type_dist}.{epochs}-{lr}-{bs}-{wxh}-{testing_dataset}'
        write_cnn_result(base_file_name, preds, ys)

