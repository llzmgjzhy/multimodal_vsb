import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import os
import builtins
import sys
import logging
from openpyxl import Workbook, load_workbook
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger("__main__")


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




def itr_test_result(config):
    """
    Calculate one iteration's test result for anomaly detection.

    New features:
    - Per-fold: find best threshold on that fold's val set, then evaluate on that fold's test set.
    - Save per-fold thresholds and metrics.
    - Keep previous global-merged functionality.
    """

    def load_one_fold(prefix: str, fold_idx: int):
        path = os.path.join(config.pred_dir, f"{prefix}_pred_{fold_idx}.csv")
        df = pd.read_csv(path)

        # safety checks
        if "pred" not in df.columns or "targets" not in df.columns:
            raise ValueError(
                f"Missing required columns in {path}. "
                f"Found columns: {list(df.columns)}; required: ['pred','targets']"
            )

        probs = df["pred"].values.astype(float)
        labels = df["targets"].values.astype(int)
        sample_ids = df["sample_id"].values.astype(int)
        return probs, labels, sample_ids

    def load_all_folds(prefix: str):
        probs_all, labels_all, sample_ids_all = [], [], []
        for i in range(config.split_num):
            probs, labels, sample_ids = load_one_fold(prefix, i)
            probs_all.append(probs)
            labels_all.append(labels)
            sample_ids_all.append(sample_ids)
        return (
            np.concatenate(probs_all),
            np.concatenate(labels_all),
            np.concatenate(sample_ids_all),
        )

    def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float):
        y_pred = (probs > threshold).astype(int)

        mcc = matthews_correlation(y_true, y_pred).item()
        acc = accuracy_score(y_true, y_pred)

        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # AUC requires both classes present; otherwise sklearn raises.
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = np.nan

        return {
            "mcc": mcc,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "auc": auc,
        }

    def _build_merged_df(meta_data_train, sample_ids, preds):
        id_meas = meta_data_train["id_measurement"].drop_duplicates().to_numpy()
        id_meas_sample = id_meas[sample_ids]
        yp_df = pd.DataFrame(
            {
                "id_measurement": id_meas_sample.astype(int),
                "prediction": np.asarray(preds, dtype=float).reshape(-1),
            }
        )

        expanded = meta_data_train[["id_measurement", "signal_id", "target"]].copy()
        expanded["id_measurement"] = expanded["id_measurement"].astype(int)

        merged_df = expanded.merge(yp_df, on="id_measurement", how="inner")
        return merged_df

    # -------------------------
    # 1) Per-fold evaluation
    # -------------------------
    fold_rows = []
    for fold in range(config.split_num):
        val_probs, val_labels, _ = load_one_fold("val", fold)
        test_probs, test_labels, _ = load_one_fold("test", fold)

        # Find best threshold on THIS fold's val set
        best_th, best_val_mcc, _ = eval_mcc(val_labels, val_probs)

        # Evaluate test with this fold-specific threshold
        test_metrics = compute_metrics(test_labels, test_probs, best_th)

        fold_rows.append(
            {
                "fold": fold,
                "best_threshold": float(best_th),
                "best_val_mcc": float(best_val_mcc),
                "test_mcc": float(test_metrics["mcc"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_f_score": float(test_metrics["f_score"]),
                "test_auc": (
                    float(test_metrics["auc"])
                    if not np.isnan(test_metrics["auc"])
                    else np.nan
                ),
                "val_size": int(len(val_labels)),
                "test_size": int(len(test_labels)),
                "val_pos_rate": float(np.mean(val_labels)),
                "test_pos_rate": float(np.mean(test_labels)),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    # summary across folds (mean + std)
    per_fold_test_mcc_mean = float(fold_df["test_mcc"].mean())
    per_fold_test_mcc_std = (
        float(fold_df["test_mcc"].std(ddof=1)) if len(fold_df) > 1 else 0.0
    )

    per_fold_threshold_mean = float(fold_df["best_threshold"].mean())
    per_fold_threshold_std = (
        float(fold_df["best_threshold"].std(ddof=1)) if len(fold_df) > 1 else 0.0
    )

    # Save per-fold details
    fold_path = os.path.join(config.pred_dir, "fold_results.csv")
    fold_df.to_csv(fold_path, index=False)

    # -------------------------
    # 2) Global merged evaluation (measurment-level)
    # -------------------------
    all_val_probs_meas, all_val_labels, all_val_sample_ids = load_all_folds("val")
    all_test_probs_meas, all_test_labels, all_test_sample_ids = load_all_folds("test")

    global_best_th_meas, global_best_val_mcc_meas, _ = eval_mcc(
        all_val_labels, all_val_probs_meas
    )

    global_test_metrics_meas = compute_metrics(
        all_test_labels, all_test_probs_meas, global_best_th_meas
    )

    # -------------------------
    # 3) Global merged evaluation (signal-level)
    # -------------------------
    meta_data_train = pd.read_csv(
        os.path.join(config.root_path, "VSBdata", "metadata_train.csv")
    )
    val_exp = _build_merged_df(meta_data_train, all_val_sample_ids, all_val_probs_meas)
    test_exp = _build_merged_df(
        meta_data_train, all_test_sample_ids, all_test_probs_meas
    )
    global_best_th_exp, global_best_val_mcc_exp, _ = eval_mcc(
        val_exp["target"].values.astype(float),  # signal-level y
        val_exp["prediction"].values.astype(float),  # broadcasted measurement score
    )
    global_exp_test_pred = (
        test_exp["prediction"].values.astype(float) > global_best_th_exp
    ).astype(int)
    global_exp_test_mcc = matthews_correlation(
        test_exp["target"].values.astype(int), global_exp_test_pred
    ).item()
    global_exp_test_metrics = compute_metrics(
        test_exp["target"].values.astype(int), test_exp["prediction"].values.astype(float), global_best_th_exp
    )

    # -------------------------
    # 4) Logging + save summary
    # -------------------------
    logger.info(
        f"[Per-fold] test MCC mean +/- std: {per_fold_test_mcc_mean:.4f} +/- {per_fold_test_mcc_std:.4f} | "
        f"threshold mean +/- std: {per_fold_threshold_mean:.6f} +/- {per_fold_threshold_std:.6f}"
    )
    logger.info(
        f"[Global] Best threshold (measurement): {global_best_th_meas:.6f}, Best val MCC (measurement): {global_best_val_mcc_meas:.4f}, "
        f"Test MCC (measurement): {global_test_metrics_meas['mcc']:.4f}"
    )
    logger.info(
        f"[Global] Best threshold: {global_best_th_exp:.6f}, "
        f"Best val MCC: {global_best_val_mcc_exp:.4f}, "
        f"Test MCC: {global_exp_test_mcc:.4f}"
    )
    # logger.info(f"Per-fold details saved to: {fold_path}")

    result = {
        # ---- global (measurement) ----
        "global_best_threshold (measurement)": float(global_best_th_meas),
        "global_best_val_mcc (measurement)": float(global_best_val_mcc_meas),
        "global_test_mcc (measurement)": float(global_test_metrics_meas["mcc"]),
        "global_test_accuracy (measurement)": float(
            global_test_metrics_meas["accuracy"]
        ),
        "global_test_precision (measurement)": float(
            global_test_metrics_meas["precision"]
        ),
        "global_test_recall (measurement)": float(global_test_metrics_meas["recall"]),
        "global_test_f_score (measurement)": float(global_test_metrics_meas["f_score"]),
        "global_test_auc (measurement)": (
            float(global_test_metrics_meas["auc"])
            if not np.isnan(global_test_metrics_meas["auc"])
            else np.nan
        ),
        # ---- global (signal-level) ----
        "global_best_threshold": float(global_best_th_exp),
        "global_best_val_mcc": float(global_best_val_mcc_exp),
        "global_exp_test_mcc": float(global_exp_test_mcc),
        "global_exp_test_accuracy": float(global_exp_test_metrics["accuracy"]),
        "global_exp_test_precision": float(global_exp_test_metrics["precision"]),
        "global_exp_test_recall": float(global_exp_test_metrics["recall"]),
        "global_exp_test_f_score": float(global_exp_test_metrics["f_score"]),
        "global_exp_test_auc": (
            float(global_exp_test_metrics["auc"])
            if not np.isnan(global_exp_test_metrics["auc"])
            else np.nan
        ),
        # ---- per-fold summary ----
        "per_fold_test_mcc_mean": per_fold_test_mcc_mean,
        "per_fold_test_mcc_std": per_fold_test_mcc_std,
        "per_fold_threshold_mean": per_fold_threshold_mean,
        "per_fold_threshold_std": per_fold_threshold_std,
        "num_folds": int(config.split_num),
        "fold_results_path": fold_path,
    }

    # Save summary CSV (single row)
    summary_path = os.path.join(config.pred_dir, "test_results.csv")
    pd.DataFrame([result]).to_csv(summary_path, index=False)
    # logger.info(f"Summary saved to: {summary_path}")

    return result


if __name__ == "__main__":
    # Example usage
    class Config:
        root_path = r"E:\Graduate\projects\multimodal_vsb_20251208\research\code\dataset"
        pred_dir = r"E:\Graduate\projects\multimodal_vsb_20251208\research\code\tensorboard\fault_detection_vsb_2026-02-14_17-03-55_kIt\predictions"
        split_num = 5

    config = Config()
    itr_test_result(config)