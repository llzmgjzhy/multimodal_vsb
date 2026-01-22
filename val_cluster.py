"""
train the model on the vsb dataset.
"""

import logging
import sys
import pandas as pd
import numpy as np
import torch
import random
import os
import time
from tqdm import tqdm
import copy

from dataset.dataset_factory import dataloader_provider
from models import model_factory
from models.optimizer import get_optimizer
from models.loss import get_loss_module
from utils.options import Options
from running import setup, pipeline_factory, evaluate_prototype_model
from utils import utils


def main(config):
    total_epoch_time = 0
    total_start_time = time.time()

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data indices
    config.data_path = os.path.join(config.root_path, config.data_path)
    meta_data_train = pd.read_csv(os.path.join(config.data_path, "metadata_train.csv"))
    query_id = "signal_id" if config.phase_level else "id_measurement"
    positive_ids = set(meta_data_train.loc[meta_data_train["target"] == 1, query_id])
    signals_ids = meta_data_train[query_id].unique()
    labels = np.array([int(id in positive_ids) for id in signals_ids], dtype=np.int64)

    # train : val : test , 3:1:1
    splits = utils.stratified_train_val_test_split(
        signals_ids, labels, num_folds=config.split_num, seed=config.seed
    )

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):

        # build data
        train_loader, val_loader, test_loader = dataloader_provider(
            config, train_idx, val_idx, test_idx, signals_ids, labels
        )

        # load model
        model_class = model_factory[config.model_name]
        model = model_class(config).to(device)

        # Evaluate on validation before training
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "./tensorboard",
                    config.cluster_dir,
                    "checkpoints",
                    f"model_best_fold_{fold_i}.pth",
                ),
                weights_only=True,
            )["state_dict"]
        )

        evaluate_prototype_model(
            model,
            val_loader,
            device,
        )

    total_runtime = time.time() - total_start_time
    print(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(total_runtime)
        )
    )


if __name__ == "__main__":
    args = Options().parse()  # `argparse` object
    origin_comment = args.comment
    for ii in range(args.itr):
        args_itr = copy.deepcopy(args)  # prevent itr forloop to change output_dir
        args_itr.seed = args_itr.seed + ii  # change seed for each iteration
        config = args  # save experiment files itr times
        config.comment = origin_comment + f" itr{ii}"
        main(config)
