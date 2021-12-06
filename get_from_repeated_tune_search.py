"""
Script to get the value of the best hyper params from the tune of each CNN
"""
import pandas as pd


def get_params(file_name, params):
    df_mean = pd.read_csv(f"tune_results/{file_name}.csv")

    min_loss_avg = df_mean["loss"].min()
    df_param = df_mean.loc[df_mean["loss"] == min_loss_avg]

    df_all = pd.read_csv(f"tune_results/{file_name}_all.csv")
    for param in params:
        value = df_param.iloc[0][param]
        df_all = df_all.loc[df_all[param] == value]

    min_loss = df_all["loss"].min()
    best_seed = df_all.loc[df_all["loss"] == min_loss].iloc[0]["trial_num"]

    return df_param[params], best_seed


if __name__ == '__main__':
    params = [
            "lr",
            "batch_size",
            "epoch",
            "wxh-stride"
        ]

    params, best_seed = get_params("kalman/alexnet", params)

    print("params", params)
    print("best_seed", best_seed)
