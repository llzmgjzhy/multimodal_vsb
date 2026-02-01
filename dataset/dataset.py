from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import pickle
import torch


class VSBTrainDataset(Dataset):
    def __init__(self, signal_ids, labels, data_path, phase_level):
        self.signal_ids = signal_ids
        self.labels = labels
        self.signal_path = os.path.join(data_path, "all_chunk_waves_160chunks.dat")
        self.phase_level = phase_level
        with open(self.signal_path, "rb") as f:
            self.data = pickle.load(f)
            # cause the data is small enough, we load all data into memory
        # Ensure data arrays are float32 to save memory and avoid conversions later
        self.data = np.asarray(self.data, dtype=np.float32)

        self.data_path = data_path
        self.phase_level = phase_level

    def __getitem__(self, index):
        # check img path exists
        # if not os.path.exists(self.img_path):
        #     print(f"Warning: Path {self.img_path} does not exist.")
        #     return None
        # signal = np.load(
        #     os.path.join(self.img_path, f"signals_{self.signal_ids[index]}.npy")
        # ).astype(np.float32)
        # signal = np.transpose(signal, (1, 0))  # [800000, 3]
        if self.phase_level:
            signal = self.data[
                self.signal_ids[index]
            ]  # signal shape is [160, 30], astype float32
        else:
            start_idx = index * 3
            end_idx = start_idx + 3
            if end_idx > len(self.data):
                raise IndexError(
                    f"Index {index} out of range: start_idx={start_idx} exceeds data length {len(self.data)}"
                )

            signal = self.data[start_idx:end_idx]  # 形状：[3, 160, 30]

        label = self.labels[index]
        return (signal, label)

    def __len__(self):
        return len(self.signal_ids)


class VSBImageDataset(Dataset):
    """
    读取你生成的两张图：
      - heatmap: global norm 图（表达幅值/分布）
      - overlay: shape 图（表达形状）
    返回：
      ((img_heat, img_over), label)
    其中 img_* 是 torch.FloatTensor, shape [C,H,W]，默认 C=3（RGB）
    """

    def __init__(
        self,
        signal_ids,
        labels,
        data_path,  # 你的 out_dir，比如 "./vsb_images"
        phase_level,
    ):
        self.signal_ids = signal_ids
        self.labels = labels
        self.img_root = Path(data_path)
        self.phase_level = phase_level

        self.heat_dir = self.img_root / "heatmap"
        self.over_dir = self.img_root / "overlay"

        if not self.heat_dir.exists():
            raise FileNotFoundError(f"heatmap dir not found: {self.heat_dir}")
        if not self.over_dir.exists():
            raise FileNotFoundError(f"overlay dir not found: {self.over_dir}")

    def __getitem__(self, index):
        sid = int(self.signal_ids[index])
        label = float(self.labels[index])

        # 你的命名是 000123.png 这种
        fname = f"{sid:06d}.png"
        img_heat = Image.open(self.heat_dir / fname).convert("RGB")
        img_over = Image.open(self.over_dir / fname).convert("RGB")

        # 注意：这里返回的是 PIL Image
        return (img_heat, img_over), label

    def __len__(self):
        return len(self.signal_ids)


class VSBImageCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):
        # batch: [((pil_heat, pil_over), label), ...]
        heats = [b[0][0] for b in batch]  # list of PIL
        overs = [b[0][1] for b in batch]  # list of PIL
        labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)

        x_heat = self.image_processor(images=heats, return_tensors="pt")["pixel_values"]
        x_over = self.image_processor(images=overs, return_tensors="pt")["pixel_values"]
        return (x_heat, x_over), labels
