"""Critical Difference Diagram custom plot.

For each evaluated metric, builds a (datasets x solvers) score table from the
benchopt run and renders a Critical Difference Diagram (Demsar, 2006). Average
ranks of each solver across datasets are plotted on a number line; horizontal
"cliques" join solvers whose pairwise difference is not statistically
significant (Wilcoxon sign-rank test with Holm correction by default).

The diagram is rendered with ``aeon.visualisation.plot_critical_difference``
(aeon is already an objective-level requirement), then converted to a numpy
RGB array — the data format expected by benchopt's ``image`` plot type.

References
----------
- benchopt custom plots:
  https://benchopt.github.io/stable/user_guide/add_custom_plot.html
- CriticalDifferenceDiagrams.jl:
  https://mirkobunse.github.io/CriticalDifferenceDiagrams.jl/dev/
- Demsar 2006, "Statistical Comparisons of Classifiers over Multiple Data
  Sets".
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchopt.plotting.base import BasePlot  # noqa: E402


# Metrics for which a smaller value indicates a better model. Used to
# auto-orient the diagram when ``lower_better`` is left to ``"auto"``.
_LOWER_BETTER_METRICS = {"mae", "mse", "rmse", "mase", "smape"}


def _fig_to_array(fig, dpi=150):
    """Render a matplotlib Figure to a (H, W, 3) array in [0, 1].

    Uses ``savefig(..., bbox_inches='tight')`` so artists drawn outside the
    axes box (e.g. aeon's solver name labels at xlim < 0.1 / > 0.9) are not
    clipped.
    """
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"))
    return img.astype(np.float32) / 255.0


def _pad_to_square(arr):
    """Pad a (H, W, 3) array with white so the result is square. Centres the
    content so it sits nicely inside benchopt's 1:1 image cell."""
    H, W, _ = arr.shape
    side = max(H, W)
    if H == W:
        return arr
    canvas = np.ones((side, side, 3), dtype=arr.dtype)
    y0 = (side - H) // 2
    x0 = (side - W) // 2
    canvas[y0:y0 + H, x0:x0 + W] = arr
    return canvas


def _text_image(message, width=8, height=8):
    """Fallback image used when a CD diagram cannot be drawn."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", fontsize=12, wrap=True,
    )
    fig.tight_layout()
    img = _fig_to_array(fig)
    plt.close(fig)
    return img


def _build_score_matrix(df, objective_column):
    """Aggregate the long-form benchopt dataframe into a (datasets, solvers)
    score matrix. Each cell holds the median across repetitions of the final
    ``stop_val`` for the given metric."""
    rows = []
    for (dataset, solver), grp in df.groupby(["dataset_name", "solver_name"]):
        last = grp.loc[
            grp["stop_val"] == grp["stop_val"].max(), objective_column
        ]
        if last.empty:
            continue
        val = float(np.nanmedian(last.to_numpy(dtype=float)))
        if np.isnan(val):
            continue
        rows.append((dataset, solver, val))

    if not rows:
        return None, None

    table = (
        pd.DataFrame(rows, columns=["dataset", "solver", "value"])
        .pivot(index="dataset", columns="solver", values="value")
        .dropna(axis=0, how="any")
    )
    if table.empty:
        return None, None
    return table.to_numpy(), table.columns.tolist()


def _resolve_lower_better(objective_column, lower_better):
    if lower_better == "True":
        return True
    if lower_better == "False":
        return False
    metric = objective_column
    if metric.startswith("objective_"):
        metric = metric[len("objective_"):]
    return metric in _LOWER_BETTER_METRICS


class Plot(BasePlot):
    """Critical Difference Diagram — one figure per (metric, orientation)."""

    name = "critical_difference_diagram"
    type = "image"
    options = {
        "objective_column": ...,
        "lower_better": ["auto", "True", "False"],
    }

    def plot(self, df, objective_column, lower_better):
        from pandas.api.types import is_numeric_dtype

        if objective_column not in df.columns:
            return [{
                "image": _text_image(
                    f"Column '{objective_column}' not in results."
                ),
                "label": objective_column,
            }]
        if not is_numeric_dtype(df[objective_column]):
            return [{
                "image": _text_image(
                    f"Column '{objective_column}' is not numeric."
                ),
                "label": objective_column,
            }]

        scores, labels = _build_score_matrix(df, objective_column)
        if (
            scores is None
            or scores.shape[0] < 2
            or scores.shape[1] < 2
        ):
            return [{
                "image": _text_image(
                    "A Critical Difference Diagram requires at least 2 "
                    "datasets and 2 solvers with overlapping results."
                ),
                "label": objective_column,
            }]

        try:
            from aeon.visualisation import plot_critical_difference
        except ImportError:
            return [{
                "image": _text_image(
                    "aeon is required for the Critical Difference Diagram."
                ),
                "label": objective_column,
            }]

        lower = _resolve_lower_better(objective_column, lower_better)
        try:
            fig, _ = plot_critical_difference(
                scores=scores,
                labels=labels,
                lower_better=lower,
                width=9,
            )
        except Exception as exc:  # noqa: BLE001
            return [{
                "image": _text_image(
                    f"plot_critical_difference failed:\n{exc}"
                ),
                "label": objective_column,
            }]

        img = _fig_to_array(fig)
        plt.close(fig)
        # benchopt's HTML JS forces each image cell into a 1:1 aspect-ratio
        # box (and uses object-contain), so a wide CD diagram would render
        # at half the cell height with a lot of empty space. Pad to square
        # so it fills the cell while preserving the original layout/labels.
        img = _pad_to_square(img)
        return [{"image": img, "label": objective_column}]

    def get_metadata(self, df, objective_column, lower_better):
        metric = objective_column
        if metric.startswith("objective_"):
            metric = metric[len("objective_"):]
        lower = _resolve_lower_better(objective_column, lower_better)
        direction = "lower is better" if lower else "higher is better"

        scores, labels = _build_score_matrix(df, objective_column) \
            if objective_column in df.columns else (None, None)
        if scores is not None:
            n_datasets, n_solvers = scores.shape
            subtitle = f"{n_datasets} datasets x {n_solvers} solvers"
        else:
            subtitle = "no overlapping results"

        return {
            "title": (
                f"Critical Difference Diagram - {metric} ({direction})\n"
                f"{subtitle}"
            ),
            "ncols": 1,
        }
