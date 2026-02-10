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
        sample_id = self.signal_ids[index]
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
        return (signal, label, sample_id)

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
        sample_id = self.signal_ids[index]
        if self.phase_level:
            sid = int(self.signal_ids[index])
            label = float(self.labels[index])

            # 你的命名是 000123.png 这种
            fname = f"{sid:06d}.png"
            img_heat = Image.open(self.heat_dir / fname).convert("RGB")
            img_over = Image.open(self.over_dir / fname).convert("RGB")
        else:
            # signal_id 是 measurement 级别的，命名是 000123_0.png, 000123_1.png, 000123_2.png
            mid = int(self.signal_ids[index])
            label = float(self.labels[index])

            img_heat_list = []
            img_over_list = []
            for chunk_i in range(3):
                frame_id = mid * 3 + chunk_i
                fname = f"{frame_id:06d}.png"
                img_heat_list.append(
                    Image.open(self.heat_dir / fname).convert("RGB")
                )  # PIL Image
                img_over_list.append(
                    Image.open(self.over_dir / fname).convert("RGB")
                )  # PIL Image

            # 将三张图拼接成一张大图，或者你也可以返回一个 list，让 dataloader 的 collate_fn 来处理
            # 这里我们简单地返回一个 list，collate_fn 会把它们堆叠成一个 batch
            img_heat = img_heat_list  # list of 3 PIL Images
            img_over = img_over_list  # list of 3 PIL Images

        # 注意：这里返回的是 PIL Image
        return (img_heat, img_over), label, sample_id

    def __len__(self):
        return len(self.signal_ids)


class VSBImageCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):
        # default measurement-level batch: list of ((pil_heat, pil_over), label, sample_id)

        # batch: [((pil_heat, pil_over), label), ...]
        heats = [b[0][0] for b in batch]  # list of PIL
        overs = [b[0][1] for b in batch]  # list of PIL
        B = len(batch)
        K = len(heats[0])  # should be 3
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        sample_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)

        # flatten to (B*K) PILs for processor
        heat_flat = [img for bag in heats for img in bag]
        over_flat = [img for bag in overs for img in bag]

        x_heat = self.image_processor(images=heat_flat, return_tensors="pt")[
            "pixel_values"
        ]
        x_over = self.image_processor(images=over_flat, return_tensors="pt")[
            "pixel_values"
        ]
        # reshape to [B,K,C,H,W]
        # x_* is [B*K, C, H, W]
        x_heat = x_heat.view(B, K, *x_heat.shape[1:])
        x_over = x_over.view(B, K, *x_over.shape[1:])
        return (x_heat, x_over), labels, sample_ids
