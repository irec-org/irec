import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    dirname(__file__) + "/data/general/recallexperiment.txt",
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
    "OurMethod2": "WSPB",
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
