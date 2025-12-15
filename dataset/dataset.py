from torch.utils.data import Dataset
import numpy as np
import os
import pickle


class VSBTrainDataset(Dataset):
    def __init__(self, signal_ids, labels, data_path):
        self.signal_ids = signal_ids
        self.labels = labels
        self.signal_path = os.path.join(data_path, "all_chunk_waves_160chunks.dat")
        with open(self.signal_path, "rb") as f:
            self.data = pickle.load(f)
            # cause the data is small enough, we load all data into memory
        # Ensure data arrays are float32 to save memory and avoid conversions later
        self.data = np.asarray(self.data, dtype=np.float32)

        self.data_path = data_path

    def __getitem__(self, index):
        # check img path exists
        # if not os.path.exists(self.img_path):
        #     print(f"Warning: Path {self.img_path} does not exist.")
        #     return None
        # signal = np.load(
        #     os.path.join(self.img_path, f"signals_{self.signal_ids[index]}.npy")
        # ).astype(np.float32)
        # signal = np.transpose(signal, (1, 0))  # [800000, 3]
        signal = self.data[
            self.signal_ids[index]
        ]  # signal shape is [160, 30], astype float32

        label = self.labels[index]
        return (signal, label)

    def __len__(self):
        return len(self.signal_ids)
