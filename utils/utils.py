import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def matthews_correlation(y_true, y_pred):
    """Calculates the Matthews correlation coefficient measure for quality of binary classification problems."""
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.float32)

    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = torch.round(torch.clamp(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = torch.sum(y_pos * y_pred_pos)
    tn = torch.sum(y_neg * y_pred_neg)

    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + torch.finfo(torch.float32).eps)


def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob, show=False):
    """
    A fast implementation of Anokas mcc optimization code.

    This code takes as input probabilities, and selects the threshold that
    yields the best MCC score. It is efficient enough to be used as a
    custom evaluation function in xgboost

    Source: https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc
    Source: https://www.kaggle.com/c/bosch-production-line-performance/forums/t/22917/optimising-probabilities-binary-prediction-script
    Creator: CPMP
    """
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = (prev_proba + proba) / 2.0 if prev_proba >= 0 else proba
            prev_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_correlation(y_true, y_pred)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_proba, best_mcc, None

def stratified_train_val_test_split(X, y, num_folds=5, seed=42):
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # get the indices of the folds
    splits = np.zeros_like(y, dtype=int)
    for fold_id, (_, val_idx) in enumerate(skf.split(X, y)):
        splits[val_idx] = fold_id

    fold_splits = []
    for val_fold in range(num_folds):
        test_fold = (val_fold + 1) % num_folds
        train_folds = [f for f in range(num_folds) if f not in [val_fold, test_fold]]

        train_idx = np.where(np.isin(splits, train_folds))[0]
        val_idx = np.where(splits == val_fold)[0]
        test_idx = np.where(splits == test_fold)[0]

        fold_splits.append((train_idx, val_idx, test_idx))

    return fold_splits

