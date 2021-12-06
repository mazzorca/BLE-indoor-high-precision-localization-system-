"""
Script to train and test a cnn
"""
from train_cnns import train_model
from cnn_testing import cnn_test, write_cnn_result, load_model
from Configuration import cnn_config
from get_from_repeated_tune_search import get_params


use_best_hyper = 1  # Use or not the best hyper parameter found with tune_cnns.py
skip_training = 0  # Do not train the model
skip_testing = 0  # Do not test the model


if __name__ == '__main__':
    model_name = "ble"  # model to be taken
    kalman = "kalman"  # use or not kalman
    transform = cnn_config.MODELS[model_name]["transform"]  # transformation to be used
    model = cnn_config.MODELS[model_name]["model"]

    # manual configuration of the hyper parameters parameters
    params = {
        "wxh-stride": "20x20-10",
        "epoch": 20,
        "batch_size": 32,
        "lr": 0.01
    }
    
    dataset = "BLE2605r"  # dataset to be used as testing

    # reproducibility of the training
    best_seed = -1
    if use_best_hyper:
        df_params, best_seed = get_params(f"{kalman}/{model_name}", list(params.keys()))
        for param in params.keys():
            params[param] = df_params.iloc[0][param]

    print("params used:", params)
    print("seed used:", best_seed)

    # training
    model_name = f"{model_name}_{kalman}"
    if not skip_training:
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

    # saving model following the naming convention
    parameters_saved = f"{model_name}/{int(params['epoch'])}-{params['lr']}-{int(params['batch_size'])}-{params['wxh-stride']}"
    model = load_model(model, parameters_saved)

    testing_datasets = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]  # dataset on which test the model

    if not skip_testing:
        number_argmax_list = [elem+1 for elem in range(18)]  # number of different square to take for testing
        for type_dist in [0, 1]:  # the two types of performances
            print("Type_dist:", type_dist)
            for testing_dataset in testing_datasets:
                for number_argmax in number_argmax_list:
                    print("testing on:", testing_dataset, "Arg for distance:", number_argmax)

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

                    base_file_name = f"cnn_results/{model_name}/{type_dist_save}.{int(params['epoch'])}-{params['lr']}-{int(params['batch_size'])}-{params['wxh-stride']}-{testing_dataset}"
                    write_cnn_result(base_file_name, preds, ys)
