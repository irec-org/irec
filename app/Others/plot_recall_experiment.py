import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    "size": 22
}

matplotlib.rc("font", **font)


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


df = pd.read_csv(
    dirname(__file__) + "/data/general/recall_experiment.txt",
    delim_whitespace=True,
    header=None,
)
recalls = list(np.around(np.arange(0.1, 0.6, 0.1), 2))
df.columns = [
    "dataset",
    "metric",
    "method",
    *recalls,
]

print(df.head(3))

print(df["metric"].unique())

print(df["method"].unique())
datasets = df["dataset"].unique()
metrics = df["metric"].unique()
methods = df["method"].unique()
markers = ["*", "x", "d", "8"]
fig, axs = plt.subplots(len(datasets), len(metrics), figsize=(17, 25))
metrics_pretty = {
    "Hits": "Hits",
    "NumInteractions": "Number of Interactions",
    "Recall": "Recall",
}

methods_pretty = {
    "GLM_UCB": "GLM-UCB",
    "LinearUCB": "Linear UCB",
    "MostPopular": "Popular",
    "WSPB": "WSPB",
}
# print(df.loc[df["method"] == "GLM_UCB"])
for i, dataset in enumerate(datasets):
    for j, metric in enumerate(metrics):
        axs[i, j].set_title(dataset)
        axs[i, j].set_xlabel("Recall")
        axs[i, j].set_ylabel(metrics_pretty[metric])
        axs[i, j].set_xticks(recalls)
        for z, method in enumerate(methods):
            # print(dataset, metric, method)
            # print(
            #     df.loc[
            #         (df.metric == metric)
            #         & (df.dataset == dataset)
            #         & (df.method == method)
            #     ][recalls].to_numpy()
            # )
            axs[i, j].plot(
                recalls,
                df.loc[
                    (df.metric == metric)
                    & (df.dataset == dataset)
                    & (df.method == method)
                ][recalls].to_numpy()[0],
                label=methods_pretty[method],
                marker=markers[z],
            )

fig.legend(
    list(map(lambda x: methods_pretty[x], methods)),
    loc="upper center",
    ncol=4,
)
fig.savefig(
    dirname(__file__) + "/data/general/recall_experiment.png",
    bbox_inches="tight",
    pad_inches=0,
)
# plt.show()

for i, dataset in enumerate(datasets):
    for j, metric in enumerate(metrics):

        fig, ax = plt.subplots()

        ax.set_title(dataset)
        ax.set_xlabel("Recall")
        ax.set_ylabel(metrics_pretty[metric])
        ax.set_xticks(recalls)
        lines = []
        for z, method in enumerate(methods):
            (line,) = ax.plot(
                recalls,
                df.loc[
                    (df.metric == metric)
                    & (df.dataset == dataset)
                    & (df.method == method)
                ][recalls].to_numpy()[0],
                label=methods_pretty[method],
                marker=markers[z],
                markersize=8,
            )
            lines.append(line)

        fig.savefig(
            dirname(__file__)
            + "/data/general/recall_experiment_num_interactions_{}_{}.png".format(
                methods_pretty[method], metrics_pretty[metric]
            ),
            bbox_inches="tight",
            pad_inches=0,
        )
        fig.savefig(
            dirname(__file__)
            + "/data/general/recall_experiment_num_interactions_{}_{}.eps".format(
                methods_pretty[method], metrics_pretty[metric]
            ),
            bbox_inches="tight",
            pad_inches=0,
        )

figlegend = plt.figure()
figlegend.legend(
    lines,
    list(map(lambda x: methods_pretty[x], methods)),
    loc="center",
    ncol=4,
)
figlegend.savefig(
    dirname(__file__) + "/data/general/recall_experiment_num_interactions_legend.png",
    bbox_inches="tight",
)
figlegend.savefig(
    dirname(__file__) + "/data/general/recall_experiment_num_interactions_legend.eps",
    bbox_inches="tight",
)
# export_legend(
#     legend,
#     filename=dirname(__file__)
#     + "/data/general/recall_experiment_num_interactions_legend.png",
# )
