from torch.utils.data import Dataset
import numpy as np
import os
import pickle


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
