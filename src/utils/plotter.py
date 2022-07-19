import numpy as np
import matplotlib.pyplot as plt


def plot_top_values(stats, n_top=25, title="", xlabel="", ylabel="",
                    normalizer=1):
    """
    Barplots the top values in a given dictionary.

    Parameters
    ----------
    stats: dict
        Dictionary of pairs string: number
    n_top: int, optional (default=25)
        Maximun number of bins
    title: str, optional (default="")
        Title for the figure
    xlabel: str, optional (default="")
        xlabel for the figure
    ylabel: str, optional (default="")
        ylabel for the figure
    normalizer: int, optional (default=1)
        Values in dictionary are divided by normalized before plotting.
    """

    # Sort by decreasing number of occurences
    sorted_stats = sorted(stats.items(), key=lambda item: -item[1])
    hot_tokens, hot_values = zip(*sorted_stats[n_top::-1])
    y_pos = np.arange(len(hot_tokens))

    # Plot
    plt.figure()
    plt.barh(hot_tokens, hot_values, align='center', alpha=0.4)
    plt.yticks(y_pos, hot_tokens, fontsize='xx-small')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show(block=False)

    return


def plot_doc_scores(y, path2figure=None):
    """
    Plot sorted document scores
    """

    breakpoint()

    plt.figure()
    plt.plot(sorted(y))
    plt.title('Sorted document scores')
    plt.xlabel('Document')
    plt.ylabel('Score')
    plt.show(block=False)

    if path2figure is not None:
        plt.savefig(path2figure)

    plt.figure()
    plt.semilogx(-np.sort(-y))
    plt.title('Sorted document scores (log-scale, descending order)')
    plt.xlabel('Document')
    plt.ylabel('Score')
    plt.show(block=False)

    if path2figure is not None:
        plt.savefig(path2figure)

    return


def plot_roc(fpr, tpr, label="", path2figure=None):
    """
    Plots a ROC curve from two lists of fpr and tpr values

    Parameters
    ----------
    fpr : array-like
        False positive rate values
    tpr : array-like
        True positive rate values
    label : str, optional (default="")
        Label for the plot
    path2figure : str or pathlib.Path or None
        Path to save the figure. If None, the figure is not saved
    """

    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, lw=2.5, label=label)
    plt.grid(b=True, which='major', color='gray', alpha=0.6,
             linestyle='dotted', lw=1.5)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC curve')
    plt.legend()

    if path2figure is not None:
        plt.savefig(path2figure)

    return


