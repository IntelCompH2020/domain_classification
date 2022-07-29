"""
A collection of methods for the evaluation of classifiers.

@author: J. Cid-Sueiro, A. Gallardo-Antolin
"""

import numpy as np

# Some libraries required for evaluation
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt


def binary_metrics(preds, labels):
    """
    Compute performance metrics based on binaray labels and predictions only

    Parameters
    ----------
    preds : np.array
        Binary predictions
    labels : np.array
        True class labels

    Returns
    -------
    eval_scores: dict
        A dictionary of evaluation metrics.
    """

    eps = 1e-50

    # Metrics computation at s_min threshold
    tn, fp, fn, tp = confusion_matrix(preds, labels).ravel()
    tpr = (tp + eps) / (tp + fn + 2 * eps)
    fpr = (fp + eps) / (fp + tn + 2 * eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + 2 * eps)
    bal_acc = 0.5 * (tpr + 1 - fpr)

    # Dictionary with the evaluation results
    m = {'size': len(labels),
         'n_labels_0': np.sum(labels == 0),
         'n_labels_1': np.sum(labels == 1),
         'n_preds_0': np.sum(preds == 0),
         'n_preds_1': np.sum(preds == 1),
         'tn': int(tn),
         'fp': int(fp),
         'fn': int(fn),
         'tp': int(tp),
         'acc': float(acc),
         'bal_acc': float(bal_acc),
         'tpr': float(tpr),
         'fpr': float(fpr)}

    return m


def print_binary_metrics(m, title=""):
    """
    Pretty-prints the given metrics

    Parameters
    ----------
    m : dict
        Dictionary of metrics (produced by the binary_metrics() method)
    title : str, optional (default="")
        Title to print as a header
    """

    print(f"")
    print("-" * len(title))
    print(title)
    print("-" * len(title))

    print(f"")
    print(f".. .. Sample size: {m['size']}")
    print(f".. .. Class proportions:")
    print(f".. .. .. Labels 0:      {m['n_labels_0']}")
    print(f".. .. .. Labels 1:      {m['n_labels_1']}")
    print(f".. .. .. Predictions 0: {m['n_preds_0']}")
    print(f".. .. .. Predictions 1: {m['n_preds_1']}")
    print(f"")
    print(f".. .. Hits:")
    print(f".. .. .. TP: {m['tp']},    TPR: {m['tpr']:.6f}")
    print(f".. .. .. TN: {m['tn']}")
    print(f".. .. Errors:")
    print(f".. .. .. FP: {m['fp']},    FPR: {m['fpr']:.6f}")
    print(f".. .. .. FN: {m['fn']}")
    print(f".. .. Standard metrics:")
    print(f".. .. .. Accuracy: {m['acc']}")
    print(f".. .. .. Balanced accuracy: {m['bal_acc']}")


def score_based_metrics(scores, labels):
    """
    Computes score-based metrics

    Parameters
    ----------
    scores : np.array
        Score values
    labels : np.array
        Target values

    Returns
    -------
    eval_scores: dict
        A dictionary of evaluation metrics.
    """

    # Sort scores and target values
    s = np.array(scores)
    ssort = -np.sort(-s)
    isort = np.argsort(-s)
    target_sorted = labels[isort]

    # ROC curve
    fpr_roc, tpr_roc, thresholds = roc_curve(target_sorted, ssort)

    # Dictionary with the evaluation results
    tpr_roc_float = [float(k) for k in tpr_roc]
    fpr_roc_float = [float(k) for k in fpr_roc]

    m = {'tpr_roc': tpr_roc_float,
         'fpr_roc': fpr_roc_float,
         'auc': auc(fpr_roc_float, tpr_roc_float)}

    return m


def plot_score_based_metrics(scores, labels):

    RocCurveDisplay.from_predictions(
        labels, scores, sample_weight=None, drop_intermediate=True,
        pos_label=None, name=None, ax=None)

    plt.show(block=False)

    return