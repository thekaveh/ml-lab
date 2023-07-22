import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from torch.utils.data import DataLoader

from common.nn.nn_model import NNModel
from common.nn.dataset.nn_dataset import NNDataset
from common.nn.params.nn_checkpoint import NNCheckpoint

sns.set()

class VisUtils:
    DEFAULT_VAL_FIG_SIZE = (25, 20)
    DEFAULT_VAL_TITLE_SIZE = 15
    DEFAULT_VAL_LABEL_SIZE = 12

    @staticmethod
    def multi_line_plot(fig_size, title, x_axis_label, x, y_axis_label, yss_legend, yss, x_ticks_inc=20, title_size=15, label_size=12):
        _ = plt.figure(figsize = fig_size)
        
        ls = ["-", "--", "-.", ":"]
        cs = sns.color_palette(n_colors=len(yss))

        for ys_idx, (ys, ys_legend) in enumerate(zip(yss, yss_legend)):
            for y_idx, y in enumerate(ys):
                ax = sns.lineplot(
                    x=x
                    , y=y
                    , color=cs[ys_idx]
                    , linestyle=ls[y_idx]
                    , label=ys_legend[y_idx]
                )

        ax.set_xlim(0, len(x))
        ax.set_title(title, fontsize=title_size)
        ax.set_ylabel(y_axis_label, fontsize=label_size)
        ax.set_xlabel(x_axis_label, fontsize=label_size)
        ax.set_xticks(range(0, len(x), x_ticks_inc))
        
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.show();
        
    @staticmethod
    def scatter_plot(vm, figsize=(15, 15), title_size=15, label_size=12):
        f, sub_plt = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        for t_idx, _ in enumerate(vm["ts"]["uni_vals"]):
            sub_plt.scatter(
                s=10
                , x=vm["xs-ts"][t_idx]
                , y=vm["ys-ts"][t_idx]
                , color=vm["ts"]["colors"][t_idx]
                , label=vm["ts"]["labels"][t_idx]
            )

        sub_plt.set_xticks([])
        sub_plt.set_yticks([])
        sub_plt.set_xlabel(vm["xs"]["label"], fontsize=label_size)
        sub_plt.set_ylabel(vm["ys"]["label"], fontsize=label_size)
        sub_plt.set_title(vm["title"], fontsize=title_size)
        sub_plt.legend(fontsize='large')
        plt.tight_layout();

    @staticmethod
    def get_scatter_plot_vm(data, title, col_xs, label_xs, col_ys, label_ys, col_ts, labels_ts, colors_ts, uni_ts):
        vm = {
            "title": title,
            "xs": {
                "vals": data[col_xs],
                "label": label_xs
            },
            "ys": {
                "vals": data[col_ys],
                "label": label_ys
            },
            "ts": {
                "vals": data[col_ts],
                "uni_vals": uni_ts,
                "colors": colors_ts,
                "labels": labels_ts
            }
        }

        vm["xs-ts"] = [vm["xs"]["vals"][vm["ts"]["vals"] == t_val] for t_val in vm["ts"]["uni_vals"]]
        vm["ys-ts"] = [vm["ys"]["vals"][vm["ts"]["vals"] == t_val] for t_val in vm["ts"]["uni_vals"]]

        return vm
            
    @staticmethod
    def visualize_checkpoint_logits_2d_tsne(
        checkpoint  : NNCheckpoint
        , ds        : NNDataset
        , n_samples : int
        , fig_size  : tuple         = DEFAULT_VAL_FIG_SIZE
        , title_size: int           = DEFAULT_VAL_TITLE_SIZE
        , label_size: int           = DEFAULT_VAL_LABEL_SIZE
    ) -> None:
        model = NNModel.from_checkpoint(checkpoint=checkpoint)
        
        ts = [t for t in range(ds.output_dim)]
        cs = sns.color_palette(n_colors=ds.output_dim)
        
        test_batch = next(iter(ds.test_loader))
        test_X, test_Y = model.net.unpack_batch(test_batch)
        test_X, test_Y = tuple(x.numpy() for x in test_X), test_Y.numpy()
        
        df_test_Y = pd.DataFrame(data=test_Y, columns=["target"])

        test_Y_hat = model.predict(X=test_X)
        
        VisUtils.scatter_plot(
            figsize=fig_size
            , title_size=title_size
            , label_size=label_size
            , vm=VisUtils.get_scatter_plot_vm(
                data=pd.concat(
                    axis=1
                    , objs=[
                        pd.DataFrame(
                            data=TSNE(
                                n_components=2
                            ).fit_transform(
                                X=pd.DataFrame(
                                    data=test_Y_hat[0]
                                ).iloc[:n_samples, :]
                            )
                            , columns=["tsne_1", "tsne_2"]
                        )
                        , df_test_Y.iloc[:n_samples, :]
                    ]
                )
                , uni_ts=ts
                , colors_ts=cs
                , labels_ts=ts
                , col_xs="tsne_1"
                , col_ys="tsne_2"
                , label_xs="tsne_1"
                , label_ys="tsne_2"
                , col_ts="target"
                , title=f"t-SNE projected 2D visualization of predicted output logits of test data snapshot @ epoch_idx={checkpoint.idp.epoch_idx}"
            )
        )