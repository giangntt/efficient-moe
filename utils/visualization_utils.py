import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matrix(matrix, title=None, xlabel=None, ylabel=None, 
                figsize=(10,8), cmap="viridis", annot=False, fmt=".2f",
                xticklabels=None, yticklabels=None):
    """
    Plot a 2D matrix as a heatmap.
    """
    # Convert to numpy if torch tensor
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    kwargs = dict(cmap=cmap, annot=annot, fmt=fmt)
    if xticklabels is not None:
        kwargs["xticklabels"] = xticklabels
    if yticklabels is not None:
        kwargs["yticklabels"] = yticklabels

    sns.heatmap(matrix, **kwargs)
    
    if title:
        plt.title(title, fontsize=16)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_bar(
    values,
    x_labels=None,
    xlabel="",
    ylabel="",
    title="",
    figsize=(8, 3),
    ylim=None,
    ax=None
):
    """
    Utility function to plot a bar chart.

    Args:
        values: list or 1D array/tensor of values
        x_labels: list of labels for x-axis (optional)
        xlabel: str, label for x-axis
        ylabel: str, label for y-axis
        title: str, plot title
        figsize: tuple, figure size
        ylim: tuple, y-axis limits (optional)
        ax: matplotlib axis to plot on (optional, if None, creates new figure)
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Convert tensor to numpy if needed
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()

    x = range(len(values))
    ax.bar(x, values)

    if x_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_histogram(values, ax=None, bins=30, color="orange", alpha=0.8, mean_line=None, mean_label=None,
                   title=None, xlabel=None, ylabel=None, xlim=None):
    """
    Utility function to plot a histogram.

    Args:
        values: list or 1D array/tensor of values
        ax: matplotlib axis to plot on (optional, if None, creates new figure)
        bins: number of bins (default: 30)
        color: color of the bars (default: "orange")
        alpha: transparency of the bars (default: 0.8)
        mean_line: mean value to draw a vertical line at (default: None)
        mean_label: label for the mean line (default: None)
        title: plot title (default: None)
        xlabel: label for x-axis (default: None)
        ylabel: label for y-axis (default: None)
        xlim: tuple of (min, max) to set the x-axis limits (default: None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(values, bins=bins, color=color, alpha=alpha)
    if mean_line is not None:
        ax.axvline(mean_line, color="red", linestyle="--", label=mean_label if mean_label else "Mean")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if mean_line is not None:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
