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
