from train_cnns import train_model, weight_reset, save_model
from cnn_testing import cnn_test, write_cnn_result, load_model, test_accuracy
from Configuration import cnn_config
from get_from_repeated_tune_search import get_params
import copy


use_best_hyper = 1


if __name__ == '__main__':
    model_name = "ble"
    transform = cnn_config.MODELS[model_name]["transform"]
    model = cnn_config.MODELS[model_name]["model"]

    params = {
        "wxh-stride": "25x25-10",
        "epoch": 20,
        "batch_size": 32,
        "lr": 0.01
    }
    dataset = "BLE2605r"

    best_seed = -1
    if use_best_hyper:
        df_params, best_seed = get_params(model_name, list(params.keys()))
        for param in params.keys():
            params[param] = df_params.iloc[0][param]

    print("params used:", params)
    print("seed used:", best_seed)
    trained_model = train_model(
        model, 
        params["wxh-stride"], 
        dataset, 
        transform, 
        int(params["epoch"]), 
        params["lr"], 
        int(params["batch_size"]), 
        model_name, 
        seed=best_seed,
        save=True
    )

    parameters_saved = f"{model_name}/{int(params['epoch'])}-{params['lr']}-{int(params['batch_size'])}-{params['wxh-stride']}"
    model = load_model(model, parameters_saved)

    testing_datasets = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]

    number_argmax_list = [4, 6, 8]
    for type_dist in [0, 1]:
        print("Type_dist:", type_dist)
        for testing_dataset in testing_datasets:
            for number_argmax in number_argmax_list:
                print("testing on:", testing_dataset)

                preds, ys = cnn_test(
                    model,
                    params["wxh-stride"],
                    testing_dataset,
                    transform,
                    int(params["batch_size"]),
                    type_dist,
                    number_argmax=number_argmax
                )

                type_dist_save = f"{type_dist}-{number_argmax}"

                base_file_name = f"cnn_results/{model_name}/{type_dist}.{int(params['epoch'])}-{params['lr']}-{int(params['batch_size'])}-{params['wxh-stride']}-{testing_dataset}"
                write_cnn_result(base_file_name, preds, ys)
