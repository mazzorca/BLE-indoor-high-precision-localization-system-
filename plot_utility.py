import matplotlib.pyplot as plt
import numpy as np


def plot_single_ax(ax, what_to_plot, ax_title="", x_label=" ", y_label=""):
    ax.set_title(ax_title)
    ax.set(xlabel=x_label, ylabel=y_label)

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(what_to_plot)))
    for i, label in enumerate(what_to_plot):
        ax.plot(what_to_plot[label], color=colors[i], linewidth=2.0, label=label)
    ax.legend()


def plot_classifier(regressor_name, data):
    fig, axs = plt.subplots(2, 2)

    plot_single_ax(axs[0, 0], data[0])
    plot_single_ax(axs[1, 0], data[1])

    plot_single_ax(axs[0, 1], data[2])
    plot_single_ax(axs[1, 1], data[3])

    fig.suptitle(regressor_name)
    plt.tight_layout()
    plt.show()
