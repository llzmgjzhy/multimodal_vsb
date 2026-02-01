from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset.dataset import VSBTrainDataset, VSBImageDataset, VSBImageCollator
from transformers import AutoImageProcessor
import os


def _create_loader(dataset_class, indices, signals_ids, labels, config, is_train=True):
    dataset = dataset_class(
        signals_ids[indices], labels[indices], config.data_path, config.phase_level
    )
    image_processor = AutoImageProcessor.from_pretrained(config.model_pretrain)
    is_shuffle = is_train
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=(
            VSBImageCollator(image_processor)
            if dataset_class == VSBImageDataset
            else None
        ),
    )


def dataloader_provider(config, train_idx, val_idx, test_idx, signals_ids, labels):
    dataset_class = (
        VSBImageDataset
        if config.data_path != os.path.join(config.root_path, "VSBdata")
        else VSBTrainDataset
    )
    train_loader = _create_loader(dataset_class, train_idx, signals_ids, labels, config)
    val_loader = _create_loader(
        dataset_class, val_idx, signals_ids, labels, config, is_train=False
    )
    test_loader = _create_loader(
        dataset_class, test_idx, signals_ids, labels, config, is_train=False
    )

    return train_loader, val_loader, test_loader
