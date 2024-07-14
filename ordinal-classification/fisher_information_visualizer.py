import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ordinal_classifier import OrdinalClassifier  # type: ignore

# set font to Arial
plt.rcParams["font.sans-serif"] = "Arial"


def ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Ordinal accuracy measures the proportion of samples whose predicted 
    label is within one of the true label.
    """

    total_count = len(y_true)

    accurate_count = sum(
        1
        for true_label, pred_label in zip(y_true, y_pred)
        if pred_label in [true_label, true_label - 1, true_label + 1]
    )
    return accurate_count / total_count


class FisherInformationVisualizer:
    def __init__(
        self,
        clf_base,
        X_train,
        y_train,
        X_test,
        y_test,
        method="t-SNE",
        perplexity=50,
    ):
        self.clf_base = clf_base
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.method = method
        self.perplexity = perplexity

        if method == "t-SNE":
            self.reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                init="random",
            )
        elif method == "PCA":
            self.reducer = PCA(n_components=2)
        else:
            raise ValueError("Method must be 't-SNE' or 'PCA'")

        self.X_train_reduced = self.reducer.fit_transform(
            np.array(X_train, dtype=np.float16)
        )
        self.X_test_reduced = self.reducer.fit_transform(
            np.array(X_test, dtype=np.float16)
        )

        self.clf_tsne = OrdinalClassifier(clf_base)
        self.clf_tsne.fit(self.X_train_reduced, y_train)

    def calculate_fim(self, dimensionality=100):
        x_min, x_max = (
            self.X_test_reduced[:, 0].min() - 1,
            self.X_test_reduced[:, 0].max() + 1,
        )
        y_min, y_max = (
            self.X_test_reduced[:, 1].min() - 1,
            self.X_test_reduced[:, 1].max() + 1,
        )
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, dimensionality),
            np.linspace(y_min, y_max, dimensionality),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        probas = self.clf_tsne.predict_proba(grid)
        log_likelihoods = np.log(np.clip(probas, 1e-15, 1 - 1e-15))

        fim = np.zeros(
            (dimensionality, dimensionality, log_likelihoods.shape[1])
        )
        for i in range(log_likelihoods.shape[1]):
            zz = log_likelihoods[:, i].reshape(xx.shape)
            gx, gy = np.gradient(zz)
            fim[:, :, i] = gx**2 + gy**2

        return fim, xx, yy

    def plot_2d_fisher_information(
        self, dimensionality=100, contour_levels=100
    ):
        fim, xx, yy = self.calculate_fim(dimensionality)
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(
            3, 3, height_ratios=[1, 1, 0.05], hspace=0.4, wspace=0.2
        )

        axes = []
        for i in range(5):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            zz = fim[:, :, i]
            contour = ax.contourf(
                xx, yy, zz, levels=contour_levels, cmap="rainbow"
            )
            ax.set_title(f"Class {i}")
            ax.set_xlabel(f"{self.method} Component 1", labelpad=5)
            ax.set_ylabel(f"{self.method} Component 2", labelpad=5)
            axes.append(ax)

        ordinal_accuracy_score = ordinal_accuracy(
            self.y_test, self.clf_tsne.predict(self.X_test_reduced)
        )
        print(f"Ordinal accuracy: {ordinal_accuracy_score:.4f}")

        cbar_ax = fig.add_subplot(gs[2, :])
        plt.colorbar(
            contour,
            cax=cbar_ax,
            orientation="horizontal",
            label="Fisher Information",
        )
        fig.suptitle(
            f"Fisher Information Maps for {type(self.clf_base).__name__} in {self.method} Space",
            fontsize=16,
            y=0.93,
        )
        plt.show()

    def plot_3d_fisher_information(self, dimensionality=100):
        # increase font during this plotting
        fim, xx, yy = self.calculate_fim(dimensionality)
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(
            3,
            3,
            height_ratios=[1, 1, 0.05],
            hspace=0.4,
            wspace=0.2,
            left=0.01,
            right=0.99,
            top=0.9,
            bottom=0.1,
        )

        axes = []
        for i in range(5):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col], projection="3d")
            zz = fim[:, :, i] / np.linalg.norm(fim[:, :, i])
            xx_flat = xx.flatten()
            yy_flat = yy.flatten()
            zz_flat = zz.flatten()
            ax.plot_trisurf(
                xx_flat, yy_flat, zz_flat, cmap="rainbow", alpha=1.0
            )

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo["tick"]["outward_factor"] = 0.0
                axis.set_pane_color((0.95, 0.95, 0.95))

            ax.set_title(f"Class {i}")
            ax.set_xlabel(f"{self.method} Component 1")
            ax.set_ylabel(f"{self.method} Component 2", labelpad=5)
            ax.set_zlabel("Information")
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.2f}")
            )
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.2f}")
            )
            ax.zaxis.set_major_formatter(
                plt.FuncFormatter(lambda z, _: f"{z:.2f}")
            )
            axes.append(ax)

        cbar_ax = fig.add_subplot(gs[2, :])
        contour = ax.contourf(xx, yy, zz, levels=100, cmap="rainbow")
        plt.colorbar(
            contour,
            cax=cbar_ax,
            orientation="horizontal",
            label="Fisher Information",
        )
        fig.suptitle(
            f"3D Fisher Information Maps for {type(self.clf_base).__name__} in {self.method} Space",
            fontsize=16,
            y=0.95,
        )
        plt.show()
