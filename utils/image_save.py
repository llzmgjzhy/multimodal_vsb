import argparse
from pathlib import Path
import pickle
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# I/O
# -----------------------------
def load_waveforms(path: Path):
    """
    读取 pickle 数据，期望返回 numpy array: [N, 160, 30]
    """
    with open(path, "rb") as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    assert arr.ndim == 3, f"waveforms should be 3D, got {arr.shape}"
    return arr


def load_indices(path: Path):
    """
    读取 fold 索引（可选），支持两种格式：
    1) pickle / npy: dict，包含 'train_idx', 'val_idx', 'test_idx'
    2) txt: 每行一个 index
    返回: np.ndarray[int]
    """
    suffix = path.suffix.lower()
    if suffix in [".pkl", ".pickle", ".dat"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            # 你可按自己的key调整
            for k in ["train_idx", "train", "trn_idx"]:
                if k in obj:
                    return np.array(obj[k], dtype=np.int64)
            raise KeyError("fold dict must contain train_idx/train/trn_idx")
        return np.array(obj, dtype=np.int64)
    if suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj.item(), dict):
            d = obj.item()
            for k in ["train_idx", "train", "trn_idx"]:
                if k in d:
                    return np.array(d[k], dtype=np.int64)
            raise KeyError("fold dict must contain train_idx/train/trn_idx")
        return np.array(obj, dtype=np.int64)
    if suffix == ".txt":
        idx = [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]
        return np.array(idx, dtype=np.int64)
    raise ValueError(f"Unsupported fold file: {path}")


# -----------------------------
# Normalization
# -----------------------------
def compute_global_robust_range(waveforms, idx=None, q_low=0.01, q_high=0.99):
    """
    waveforms: [N, 160, 30]
    idx: 可选，指定用哪些样本统计 global range（建议只用训练集 idx，避免泄漏）
    返回: (q1, q99)
    """
    if idx is None:
        x = waveforms.reshape(-1).astype(np.float32)
    else:
        x = waveforms[idx].reshape(-1).astype(np.float32)

    q1 = np.quantile(x, q_low)
    q99 = np.quantile(x, q_high)
    # 防止极端情况 q1==q99
    if np.isclose(q1, q99):
        q99 = q1 + 1e-6
    return float(q1), float(q99)


def normalize_global_robust(w, q1, q99, eps=1e-6):
    """
    w: [160, 30]
    按 global q1~q99 clip，再缩放到 [0,1]
    """
    w = w.astype(np.float32)
    w = np.clip(w, q1, q99)
    w = (w - q1) / (q99 - q1 + eps)
    return w


def normalize_shape_per_sample(w, mode="zscore", eps=1e-6):
    """
    w: [160, 30]
    用于堆叠图（突出形状，不强调幅值）
    - zscore: (w-mean)/std，形状更明显
    - minmax: per-sample 归一化到 [0,1]，也可用，但形状对比有时不如 zscore
    返回归一化后的 w，以及推荐的 y-lim
    """
    w = w.astype(np.float32)

    if mode == "minmax":
        mn, mx = w.min(), w.max()
        w2 = (w - mn) / (mx - mn + eps)
        return w2, (0.0, 1.0)

    # 默认 zscore
    m = w.mean()
    s = w.std() + eps
    w2 = (w - m) / s

    # 画堆叠图时固定 y-lim（避免每张图自动缩放引入伪特征）
    # 用 robust 范围更稳
    lo, hi = np.quantile(w2, 0.01), np.quantile(w2, 0.99)
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return w2, (float(lo), float(hi))


# -----------------------------
# Plot saving
# -----------------------------
def save_heatmap(w01, out_path: Path, img_size=224, cmap="viridis"):
    """
    w01: [160,30] 已缩放到[0,1]（global norm后的热力图）
    """
    dpi = 100
    fig = plt.figure(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")

    # vmin/vmax 固定为 [0,1]，确保颜色标尺一致
    ax.imshow(
        w01, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_overlay(
    w_shape, out_path: Path, img_size=224, ylim=None, linewidth=0.8, alpha=0.12
):
    """
    w_shape: [160,30] (per-sample zscore/minmax)
    ylim: 固定 y 轴范围（强烈建议传入）
    """
    dpi = 100
    fig = plt.figure(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")

    x = np.arange(w_shape.shape[1], dtype=np.int32)
    ax.set_xlim(0, w_shape.shape[1] - 1)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # 160条脉冲叠加
    for i in range(w_shape.shape[0]):
        ax.plot(x, w_shape[i], alpha=alpha, linewidth=linewidth,color="C0")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# -----------------------------
# Export
# -----------------------------
def export_images(
    waveforms,
    out_dir: Path,
    q1,
    q99,
    img_size=224,
    cmap="viridis",
    overlay_norm="zscore",
    every_n=1,
    indices=None,
):
    """
    waveforms: [N,160,30]
    indices: 如果不为None，只导出这些索引（比如 train/val/test 单独导出）
    every_n: 调试用，隔n个导出
    """
    out_heat = out_dir / "heatmap"
    out_over = out_dir / "overlay"
    out_heat.mkdir(parents=True, exist_ok=True)
    out_over.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = np.arange(waveforms.shape[0], dtype=np.int64)

    for c, idx in enumerate(indices[::every_n]):
        w = waveforms[idx]  # [160,30]

        # heatmap：global robust → [0,1]
        w_heat = normalize_global_robust(w, q1, q99)
        save_heatmap(w_heat, out_heat / f"{idx:06d}.png", img_size=img_size, cmap=cmap)

        # overlay：per-sample → 形状
        w_over, ylim = normalize_shape_per_sample(w, mode=overlay_norm)
        save_overlay(w_over, out_over / f"{idx:06d}.png", img_size=img_size, ylim=ylim)

        if c % 200 == 0:
            print(f"[export] {c}/{len(indices)} saved (last idx={idx})")


# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        default="E:\\Graduate\\projects\\multimodal_vsb_20251208\\research\\code\\dataset\\VSBdata",
        help="Directory containing the data files",
    )
    p.add_argument("--data_file", type=str, default="all_chunk_waves_160chunks.dat")
    p.add_argument("--out_dir", type=str, default="./dataset/VSBdata/vsb_images")

    # global norm 用训练 fold 统计，避免泄漏（推荐）
    p.add_argument(
        "--fold_train_idx_file",
        type=str,
        default=None,
        help="可选：训练集索引文件，用于统计 global q1/q99（避免val/test泄漏）",
    )

    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument(
        "--overlay_norm", type=str, default="minmax", choices=["zscore", "minmax"]
    )
    p.add_argument("--every_n", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_path = Path(args.data_dir) / args.data_file
    waveforms = load_waveforms(data_path)  # [N,160,30]

    train_idx = None
    if args.fold_train_idx_file is not None:
        train_idx = load_indices(Path(args.fold_train_idx_file))

    q1, q99 = compute_global_robust_range(
        waveforms, idx=train_idx, q_low=0.01, q_high=0.99
    )
    print(
        f"[global norm] q1={q1:.6f}, q99={q99:.6f} (idx={'train' if train_idx is not None else 'ALL'})"
    )

    export_images(
        waveforms,
        out_dir=Path(args.out_dir),
        q1=q1,
        q99=q99,
        img_size=args.img_size,
        cmap=args.cmap,
        overlay_norm=args.overlay_norm,
        every_n=args.every_n,
        indices=None,  # 或者你传 train/val/test 的 indices 做分开导出
    )

    print("Done.")
