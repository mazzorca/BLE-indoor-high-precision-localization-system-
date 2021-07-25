import matplotlib.pyplot as plt
import numpy as np


def plot_single_ax(ax, what_to_plot, ax_title="", x_label="", y_label=""):
    ax.set_title(ax_title)
    ax.set(xlabel=x_label, ylabel=y_label)

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(what_to_plot)))
    for i, label in enumerate(what_to_plot):
        ax.plot(what_to_plot[label], color=colors[i], linewidth=2.0, label=label)
    ax.legend()


def plot(plot_name, data, num_row, num_col):
    fig, axs = plt.subplots(num_row, num_col)

    if num_row == 1 or num_col == 1:
        if num_row == num_col:
            plot_single_ax(axs, data[0])
        else:
            dim = num_row if num_row != 1 else num_col
            for i in range(dim):
                plot_single_ax(axs[i], data[i])
    else:
        for j in range(num_col):
            for i in range(num_row):
                plot_single_ax(axs[i, j], data[j * num_col + i])

    fig.suptitle(plot_name)
    plt.tight_layout(pad=0)
    plt.show()
