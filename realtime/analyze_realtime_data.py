import csv
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

# runs = [
#     "realtime_data/central_points_ble_rnn_kalman/cnn_pos_central_points.csv",
#     "realtime_data/central_kalman/rnn_pos_central_points.csv",
#     "realtime_data/central_kalman/Nearest Neighbors D_pos_central_points.csv"
# ]

runs = [
    "realtime_data/central_nokalman/cnn_pos_central_points.csv",
    "realtime_data/central_nokalman/rnn_pos_central_points.csv"
]

# runs = [
#     "realtime_data/near_points_ble_rnn_kalman/cnn_pos_near_points.csv",
#     "realtime_data/near_kalman/Nearest Neighbors D_pos_near_points.csv",
#     "realtime_data/near_kalman/rnn_pos_near_points.csv"
# ]

# runs = [
#     "realtime_data/central_long/cnn_pos_central_points.csv",
#     "realtime_data/central_long/Nearest Neighbors D_pos_central_points.csv",
#     "realtime_data/central_long/rnn_pos_central_points.csv"
# ]

# runs = [
#     "realtime_data/near_long/cnn_pos_near_points.csv",
#     "realtime_data/near_long/Nearest Neighbors D_pos_near_points.csv",
#     "realtime_data/near_long/rnn_pos_near_points.csv"
# ]


def table_time(path_story, trajectory_name):
    with open(f'../{trajectory_name}.json', ) as f:
        data = json.load(f)

    name_model = path_story.split("/")[2].split("_")[0]

    df = pd.read_csv(f"../{path_story}")
    initial_time = df['time'][0]
    df['time'] = df['time'] - initial_time

    px = 1 / plt.rcParams['figure.dpi']
    ax = df.plot.scatter(
        x='x',
        y='y',
        c='time',
        colormap='turbo',
        title=f"tracement Real time {name_model}",
        xlabel="x(m)",
        ylabel="y(m)",
        figsize=(950 * px, 400 * px)
    )

    for point in data["points"]:
        ax.scatter(point['x'], point['y'], c='black')

    plt.xlim(0, 1.80)
    plt.ylim(0, 0.90)

    major_ticks_x = np.arange(0, 1.81, 0.30)
    major_ticks_y = np.arange(0, 0.91, 0.30)

    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)

    ax.grid(which='both')

    name_model = path_story.split("/")[2].split("_")[0]
    traj_name = trajectory_name.split("_")[0]
    plt.savefig(f"../plots/realtime/table_{name_model}_{traj_name}.png")


def print_lag_error(path_story, trajectory_name):
    with open(f'../{trajectory_name}.json', ) as f:
        data = json.load(f)

    df = pd.read_csv(f"../{path_story}")

    lag_list, o_list_x, o_list_y, p_list_x, p_list_y = get_errors_in_the_run(data, df)

    lag_df = pd.DataFrame({
        'Error': lag_list
    })

    lag_x = pd.DataFrame({
        'p_x': p_list_x,
        'o_x': o_list_x
    })
    lag_y = pd.DataFrame({
        'p_y': p_list_y,
        'o_y': o_list_y
    })

    # lag_df.plot.line(
    #     title="tracement Real time Error",
    #     xlabel="Seconds",
    #     ylabel="m"
    # )

    name_model = path_story.split("/")[2].split("_")[0]
    traj_name = trajectory_name.split("_")[0]
    # plt.savefig(f"../plots/realtime/error_{name_model}_{traj_name}.png")
    # plt.close()

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("tracement Real time Error x")
    ax[0].set(xlabel="Seconds", ylabel="m")

    ax[0].plot(
        p_list_x, color="tab:orange", label="predicted_x"
    )
    ax[0].plot(
        o_list_x, color="tab:blue", label="optimal_x"
    )

    for i in range(18):
        ax[0].axvline(x=i * 10, color="black", linestyle=":")

    ax[1].set_title("tracement Real time Error y")
    ax[1].set(xlabel="Seconds", ylabel="m")

    ax[1].plot(
        p_list_y, color="tab:orange", label="predicted_y"
    )
    ax[1].plot(
        o_list_y, color="tab:blue", label="optimal_y"
    )

    for i in range(18):
        ax[1].axvline(x=i * 10, color="black", linestyle=":")

    ax[0].set_xlim(0, 180)
    ax[0].legend()
    ax[1].set_xlim(0, 180)
    ax[1].legend()

    plt.tight_layout()

    plt.savefig(f"../plots/realtime/error_{name_model}_{traj_name}_x_y.png")


def get_errors_in_the_run(data, df):
    index_point = 0
    next_time = data["points"][0]["time"]
    block_transition = False
    counter_transition = 0
    lag_list = []
    p_list_x = []
    p_list_y = []
    o_list_x = []
    o_list_y = []
    transtion_points_x = []
    transtion_points_y = []
    for index, row in df.iterrows():
        if counter_transition > 0 and block_transition:
            counter_transition -= 1
            x = row["x"]
            y = row["y"]
            p_list_x.append(x)
            p_list_y.append(y)
            x_opt = transtion_points_x[counter_transition]
            y_opt = transtion_points_y[counter_transition]
            o_list_x.append(x_opt)
            o_list_y.append(y_opt)
            if counter_transition == 0:
                block_transition = False
                next_time += data["points"][index_point]["time"]
            continue

        x = row["x"]
        y = row["y"]
        p_predicted = np.array([x, y])
        p_list_x.append(x)
        p_list_y.append(y)
        p_optimal = np.array([float(data["points"][index_point]["x"]), float(data["points"][index_point]["y"])])
        x_opt = float(data["points"][index_point]["x"])
        y_opt = float(data["points"][index_point]["y"])
        o_list_x.append(x_opt)
        o_list_y.append(y_opt)
        dist = np.linalg.norm(p_predicted - p_optimal)
        lag_list.append(dist)

        if index > next_time:
            if index_point < len(data["points"]) - 1:
                index_point += 1

            transtion_points_x = np.linspace(float(data["points"][index_point]["x"]), x_opt, 5)
            transtion_points_y = np.linspace(float(data["points"][index_point]["y"]), y_opt, 5)
            next_time += 5
            block_transition = True
            counter_transition = 5
    return lag_list, o_list_x, o_list_y, p_list_x, p_list_y


def all_errors(path_storys, trajectory_name):
    with open(f'../{trajectory_name}.json', ) as f:
        data = json.load(f)

    traj_name = trajectory_name.split("_")[0]

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("tracement Real time Error x")
    ax[0].set(xlabel="Seconds", ylabel="m")

    ax[1].set_title("tracement Real time Error y")
    ax[1].set(xlabel="Seconds", ylabel="m")

    colors = ["tab:orange", "tab:green", "tab:red"]
    first = True
    for i, path_story in enumerate(path_storys):
        name_model = path_story.split("/")[2].split("_")[0]
        df = pd.read_csv(f"../{path_story}")

        lag_list, o_list_x, o_list_y, p_list_x, p_list_y = get_errors_in_the_run(data, df)

        ax[0].plot(
            p_list_x, color=colors[i], label=f"predicted x {name_model}"
        )
        if first:
            ax[0].plot(
                o_list_x, color="tab:blue", label=f"optimal x"
            )

        ax[1].plot(
            p_list_y, color=colors[i], label=f"predicted y {name_model}"
        )
        if first:
            ax[1].plot(
                o_list_y, color="tab:blue", label=f"optimal y"
            )

        first = False

    for i in range(18):
        ax[0].axvline(x=i * 10, color="black", linestyle=":")

    for i in range(18):
        ax[1].axvline(x=i * 10, color="black", linestyle=":")

    ax[0].set_xlim(0, 115)
    ax[0].legend()
    ax[1].set_xlim(0, 115)
    ax[1].legend()

    plt.tight_layout()

    plt.savefig(f"../plots/realtime/error_{traj_name}_all.png")


if __name__ == '__main__':
    for run in runs:
        table_time(run, "central_trajectory")
    # table_time("realtime_data/central_long/Nearest Neighbors D_pos_central_points.csv", "central_trajectory_fixed_time")
    # print_lag_error("realtime_data/central_long/Nearest Neighbors D_pos_central_points.csv",
    #                 "central_trajectory_fixed_time")
    all_errors(runs, "central_trajectory")
