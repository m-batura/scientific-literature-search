import numpy as np
import pandas as pd
import sklearn.metrics as skl
import matplotlib.pyplot as plt

def dot(vector1, vector2):
    return np.dot(vector1, vector2)

def single_class(group, score):
    true = (group == 1).astype(int)
    top5_idx = score.nlargest(5).index
    pred = pd.Series(0, index=score.index)
    pred.loc[top5_idx] = 1
    return true.values, score.values, pred.values

def multi_class(group, score):
    sorted_idx = score.sort_values(ascending=False).index
    pred = pd.Series(0, index=score.index)
    for g in range(3):
        start = g * 5
        end = (g + 1) * 5
        block_idx = sorted_idx[start:end]
        pred.loc[block_idx] = g + 1
    return group.values, pred.values


def accuracy(true, pred):
    return skl.accuracy_score(true, pred)

def precision(true, pred):
    return skl.precision_score(true, pred)

def recall(true, pred):
    return skl.recall_score(true, pred)

def f1(true, pred):
    return skl.f1_score(true, pred)

def plot_roc(true, score, filename=".\\results\\roc_curve.svg"):
    fpr, tpr, thresholds = skl.roc_curve(true, score)
    roc_auc = skl.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

    return fpr, tpr, thresholds

def plot_pr(true, score, filename=".\\results\\pr_curve.svg"):
    precision, recall, thresholds = skl.precision_recall_curve(true, score)
    ap = skl.average_precision_score(true, score)

    plt.figure()
    plt.plot(recall, precision, color="green", label=f"PR curve (AP = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(filename)   # stored to file
    plt.close()

    return precision, recall, thresholds

def optimal_thresholds(true, score):
    fpr, tpr, thresholds = skl.roc_curve(true, score)
    youden_j = tpr - fpr
    best_j_idx = np.argmax(youden_j)
    best_j_threshold = thresholds[best_j_idx]

    precision, recall, pr_thresholds = skl.precision_recall_curve(true, score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 1.0

    return best_j_threshold, best_f1_threshold