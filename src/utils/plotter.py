import logging
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def add_tag_2_path(tag, path):
    """
    Add a tag at the end of the name of the file in the given path

    Parameters
    ----------
    suffix : str
    pathlib.Path
        suffix to add to the filename

    path : pathlib.Path
        A file name or a path to a file
    """

    path = pathlib.Path(path)

    return path.parent / f'{path.stem}_{tag}{path.suffix}'


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


def plot_doc_scores(scores, n_pos, path2figure=None):
    """
    Plot sorted document scores

    Parameters
    ----------
    scores : list
        A list of scores

    n_pos : float
        The position to mark in the plot.
    """

    # Parameters
    N = len(scores)
    s_max = np.max(scores)

    # Sort scores in ascending order
    sorted_scores = sorted(scores)

    # #############
    # Sorted scores

    # Plot sorted scores in linear scale
    plt.figure()
    plt.plot(sorted_scores, label='score')
    # Plot score threshold
    z = N - n_pos
    plt.plot([z, z], [0, s_max], ':r', label='threshold')
    plt.title('Sorted document scores')
    plt.xlabel('Document')
    plt.ylabel('Score')
    plt.legend()
    plt.show(block=False)

    if path2figure is not None:
        plt.savefig(path2figure)
        logging.info(f"-- Figure saved in {path2figure}")
        plt.savefig(path2figure)

    # Plot sorted scores in xlog scale and descending order
    plt.figure()
    plt.semilogx(range(1, N + 1), -np.sort(-scores), label='score')
    # Plot score threshold
    plt.semilogx([n_pos, n_pos], [0, s_max], ':r', label='threshold')
    plt.title('Sorted document scores (log-scale, descending order)')
    plt.xlabel('Document')
    plt.ylabel('Score')
    plt.legend()
    plt.show(block=False)

    if path2figure is not None:
        path2figure_log = add_tag_2_path('log', path2figure)
        logging.info(f"-- Figure saved in {path2figure_log}")
        plt.savefig(path2figure_log)

    # ###############
    # Score histogram

    # Plot sorted scores in xlog scale and descending order
    plt.figure()
    # Set log=True to show bar heights in log scale.
    plt.hist(scores, bins=20, log=False)
    plt.title('Score distribution')
    plt.xlabel('Score')
    plt.ylabel('Number of items')
    plt.show(block=False)

    if path2figure is not None:
        path2figure_hist = add_tag_2_path('hist', path2figure)
        logging.info(f"-- Figure saved in {path2figure_hist}")
        plt.savefig(path2figure_hist)

    return


def base_plot_roc(fpr, tpr, fpr0=None, tpr0=None, label="", path2figure=None,
                  title='ROC curve'):
    """
    Plots a ROC curve from two lists of fpr and tpr values

    Parameters
    ----------
    fpr : array-like
        False positive rate values
    tpr : array-like
        True positive rate values
    fpr0 : float or None, optional (default=None)
        FPR of the operating point. If none no operation point is plotted
    tpr0 : float or None, optional (default=None)
        TPR of the operating point. If none no operation point is plotted
    label : str, optional (default="")
        Label for the plot
    path2figure : str or pathlib.Path or None
        Path to save the figure. If None, the figure is not saved
    """

    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, lw=2.0, label=label)
    if fpr0 is not None and tpr0 is not None:
        plt.plot([fpr0], [tpr0], '.', color='red', markersize=10,
                 label='Op. point')
    plt.grid(b=True, which='major', color='gray', alpha=0.6,
             linestyle='dotted', lw=1.5)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend()
    plt.show(block=False)

    if path2figure is not None:
        plt.savefig(path2figure)

    return


def plot_roc(roc, metrics, tag="", path2figure=None):
    """
    Plots a ROC curve from two data dictionaries.

    Parameters
    ----------
    roc : dict
        A dictionary of roc values. It should containt heys fpr_roc, tpr_roc
        and auc
    metrics : dict
        A dictionary of performance metrics. It should contain keys fpr0 and
        tpr0 (the point of operation)
    tag : str, optional (default="")
        Tipically, a label 'Train' or 'Test' to add to the figure texts
    path2figure : str or pathlib.Path or None
        Path to save the figure. If None, the figure is not saved
    """

    if roc is not None:
        base_plot_roc(
            roc['fpr_roc'], roc['tpr_roc'],
            fpr0=metrics['fpr'],
            tpr0=metrics['tpr'],
            title=f"ROC ({tag})",
            label=f"{tag} (AUC = {roc['auc']:.2f})",
            path2figure=path2figure)

    return

