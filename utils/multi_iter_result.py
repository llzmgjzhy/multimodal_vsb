import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from utils import matthews_correlation, eval_mcc

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


def _parse_ts(name: str) -> datetime:
    m = _TS_RE.search(os.path.basename(os.path.normpath(name)))
    if not m:
        raise ValueError(f"Cannot find timestamp in: {name}")
    return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")


def _resolve_output_dir(p: str) -> str:
    p = os.path.normpath(p)
    return os.path.dirname(p) if os.path.basename(p) == "predictions" else p


def _pred_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "predictions")


def _load_one(pred_dir: str, prefix: str, fold: int) -> pd.DataFrame:
    path = os.path.join(pred_dir, f"{prefix}_pred_{fold}.csv")
    df = pd.read_csv(path)
    need = {"sample_id", "pred", "targets"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"{path} missing columns. Found={list(df.columns)} need={sorted(need)}"
        )
    df = df[["sample_id", "pred", "targets"]].copy()
    df["sample_id"] = df["sample_id"].astype(int)
    df["pred"] = df["pred"].astype(float)
    df["targets"] = df["targets"].astype(int)
    return df


def _concat_all_folds(pred_dir: str, prefix: str, split_num: int) -> pd.DataFrame:
    return pd.concat(
        [_load_one(pred_dir, prefix, f) for f in range(split_num)], ignore_index=True
    )


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, th: float):
    y_pred = (probs > th).astype(int)
    mcc = matthews_correlation(y_true, y_pred).item()
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = np.nan
    return dict(
        mcc=mcc,
        accuracy=acc,
        precision=precision,
        recall=recall,
        f_score=f_score,
        auc=auc,
    )


def _build_merged_df(
    meta_data_train: pd.DataFrame, sample_ids: np.ndarray, preds: np.ndarray
) -> pd.DataFrame:
    id_meas = meta_data_train["id_measurement"].drop_duplicates().to_numpy()
    id_meas_sample = id_meas[sample_ids.astype(int)]
    yp_df = pd.DataFrame(
        {
            "id_measurement": id_meas_sample.astype(int),
            "prediction": np.asarray(preds, dtype=float).reshape(-1),
        }
    )
    expanded = meta_data_train[["id_measurement", "signal_id", "target"]].copy()
    expanded["id_measurement"] = expanded["id_measurement"].astype(int)
    return expanded.merge(yp_df, on="id_measurement", how="inner")


def ensemble_avg_pred_between_dirs(
    start_dir: str,
    end_dir: str,
    drop_overlap: bool = False,
    save_dir_name: str = "ensemble_avg_pred_between",
):
    """
    在时间窗口内找到所有实验目录：
      - 每个itr：拼全fold val/test
      - 跨itr：按sample_id平均pred
      - 用平均val找阈值、平均test算指标（measurement+signal-level）
    """

    start_out = _resolve_output_dir(start_dir)
    end_out = _resolve_output_dir(end_dir)
    parent = os.path.dirname(start_out)
    if os.path.dirname(end_out) != parent:
        raise ValueError(
            "start_dir and end_dir must be under the same parent directory"
        )

    t0, t1 = _parse_ts(start_out), _parse_ts(end_out)
    tmin, tmax = (t0, t1) if t0 <= t1 else (t1, t0)

    # 找时间窗口内 runs
    runs = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue
        m = _TS_RE.search(name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        if tmin <= ts <= tmax and os.path.isdir(_pred_dir(full)):
            runs.append((ts, full))
    if not runs:
        raise ValueError("No runs found in time window")

    runs.sort(key=lambda x: x[0])

    # 1) 收集所有itr的“全fold val/test”
    all_val_parts, all_test_parts = [], []
    for ts, out_dir in runs:
        pred_dir = _pred_dir(out_dir)
        v = _concat_all_folds(pred_dir, "val", 5)
        te = _concat_all_folds(pred_dir, "test", 5)
        v["run_time"] = ts.strftime("%Y-%m-%d_%H-%M-%S")
        te["run_time"] = ts.strftime("%Y-%m-%d_%H-%M-%S")
        all_val_parts.append(v)
        all_test_parts.append(te)

    all_val = pd.concat(all_val_parts, ignore_index=True)
    all_test = pd.concat(all_test_parts, ignore_index=True)

    # 2) 跨itr按 sample_id 平均 pred（targets 应该一致，取 first 即可）
    val_avg = all_val.groupby("sample_id", as_index=False).agg(
        pred=("pred", "mean"),
        targets=("targets", "first"),
        n=("pred", "size"),
    )
    test_avg = all_test.groupby("sample_id", as_index=False).agg(
        pred=("pred", "mean"),
        targets=("targets", "first"),
        n=("pred", "size"),
    )

    # 3) 可选：去掉 val/test 重叠 sample，避免阈值调参泄漏到 test
    if drop_overlap:
        overlap = set(val_avg["sample_id"]).intersection(set(test_avg["sample_id"]))
        if overlap:
            # 常用做法：从 test 中剔除重叠，保证 test 更“干净”
            test_avg = test_avg[~test_avg["sample_id"].isin(overlap)].reset_index(
                drop=True
            )

    # 4) measurement-level 全局：val找阈值，test算指标
    best_th_meas, best_val_mcc_meas, _ = eval_mcc(
        val_avg["targets"].values, val_avg["pred"].values
    )
    test_metrics_meas = _compute_metrics(
        test_avg["targets"].values, test_avg["pred"].values, best_th_meas
    )

    # 5) signal-level（按你原函数的 measurement->signal broadcast 方式）
    meta_data_train = pd.read_csv(
        os.path.join("./dataset", "VSBdata", "metadata_train.csv")
    )
    val_exp = _build_merged_df(
        meta_data_train, val_avg["sample_id"].values, val_avg["pred"].values
    )
    test_exp = _build_merged_df(
        meta_data_train, test_avg["sample_id"].values, test_avg["pred"].values
    )

    best_th_exp, best_val_mcc_exp, _ = eval_mcc(
        val_exp["target"].values.astype(float),
        val_exp["prediction"].values.astype(float),
    )
    exp_test_pred = (test_exp["prediction"].values.astype(float) > best_th_exp).astype(
        int
    )
    exp_test_mcc = matthews_correlation(
        test_exp["target"].values.astype(int), exp_test_pred
    ).item()

    # 6) 保存
    out_dir = os.path.join(
        parent,
        f"{save_dir_name}_{tmin.strftime('%Y-%m-%d_%H-%M-%S')}_to_{tmax.strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    val_avg.to_csv(os.path.join(out_dir, "val_avg_by_sample.csv"), index=False)
    test_avg.to_csv(os.path.join(out_dir, "test_avg_by_sample.csv"), index=False)
    pd.DataFrame({"run_output_dir": [d for _, d in runs]}).to_csv(
        os.path.join(out_dir, "runs_in_window.csv"), index=False
    )

    result = {
        "num_runs": int(len(runs)),
        "time_start": tmin.strftime("%Y-%m-%d_%H-%M-%S"),
        "time_end": tmax.strftime("%Y-%m-%d_%H-%M-%S"),
        "drop_overlap": bool(drop_overlap),
        "val_unique_samples": int(len(val_avg)),
        "test_unique_samples": int(len(test_avg)),
        "global_best_threshold (measurement)": float(best_th_meas),
        "global_best_val_mcc (measurement)": float(best_val_mcc_meas),
        "global_test_mcc (measurement)": float(test_metrics_meas["mcc"]),
        "global_test_accuracy (measurement)": float(test_metrics_meas["accuracy"]),
        "global_test_precision (measurement)": float(test_metrics_meas["precision"]),
        "global_test_recall (measurement)": float(test_metrics_meas["recall"]),
        "global_test_f_score (measurement)": float(test_metrics_meas["f_score"]),
        "global_test_auc (measurement)": (
            float(test_metrics_meas["auc"])
            if not np.isnan(test_metrics_meas["auc"])
            else np.nan
        ),
        "global_best_threshold": float(best_th_exp),
        "global_best_val_mcc": float(best_val_mcc_exp),
        "global_exp_test_mcc": float(exp_test_mcc),
        "ensemble_output_dir": out_dir,
    }
    pd.DataFrame([result]).to_csv(
        os.path.join(out_dir, "test_results.csv"), index=False
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble average predictions between two runs"
    )
    parser.add_argument(
        "--start_dir",
        type=str,
        default=r"E:\Graduate\projects\multimodal_vsb_20251208\research\code\tensorboard\fault_detection_vsb_2026-02-13_23-05-35_1tq",
        help="Start of time window (inclusive)",
    )
    parser.add_argument(
        "--end_dir",
        type=str,
        default=r"E:\Graduate\projects\multimodal_vsb_20251208\research\code\tensorboard\fault_detection_vsb_2026-02-14_06-53-10_z7y",
        help="End of time window (inclusive)",
    )
    parser.add_argument(
        "--drop_overlap",
        action="store_true",
        help="Whether to drop overlapping samples between val and test",
    )
    parser.add_argument(
        "--save_dir_name",
        type=str,
        default="ensemble_avg_pred_between",
        help="Name for the output directory",
    )
    args = parser.parse_args()

    result = ensemble_avg_pred_between_dirs(
        start_dir=args.start_dir,
        end_dir=args.end_dir,
        drop_overlap=args.drop_overlap,
        save_dir_name=args.save_dir_name,
    )
