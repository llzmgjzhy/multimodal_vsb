import logging
from datetime import datetime
import random
import string
import os
from utils import utils
import json
from collections import OrderedDict
import torch
import time
import numpy as np
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)
from utils.utils import matthews_correlation, eval_mcc, augment_pulse_set_vsb, augment_signal

logger = logging.getLogger("__main__")

NEG_METRICS = {"loss", "mse"}

val_times = {"total_time": 0, "count": 0}


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    # config = args.__dict__  # configuration dictionary

    # if args.config_filepath is not None:
    #     logger.info("Reading configuration ...")
    #     try:  # dictionary containing the entire configuration settings in a hierarchical fashion
    #         config.update(utils.load_config(args.config_filepath))
    #     except:
    #         logger.critical(
    #             "Failed to load configuration file. Check JSON syntax and verify that files exist"
    #         )
    #         traceback.print_exc()
    #         sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    if not os.path.isdir(args.output_dir):
        raise IOError(
            f"Root directory '{args.output_dir}', where the directory of the experiment will be created, must exist"
        )

    output_dir = os.path.join(args.output_dir, args.experiment_name)

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    args.initial_timestamp = formatted_timestamp
    if (not args.no_timestamp) or (len(args.experiment_name) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += f"_{formatted_timestamp}_{rand_suffix}"
    args.output_dir = output_dir
    args.save_dir = os.path.join(output_dir, "checkpoints")
    args.pred_dir = os.path.join(output_dir, "predictions")
    args.tensorboard_dir = os.path.join(output_dir, "tb_summaries")
    utils.create_dirs([args.save_dir, args.pred_dir, args.tensorboard_dir])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return args


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config.task

    if task == "fault_detection":
        return Anomaly_Detection_Runner

    if task == "classification":
        return ClassificationRunner

    if task == "cluster":
        return Cluster_Runner

    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


class BaseRunner(object):

    def __init__(
        self,
        model,
        dataloader,
        device,
        loss_module,
        config,
        optimizer=None,
        l2_reg=None,
        print_interval=10,
        console=True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.config = config
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError("Please override in child class")

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError("Please override in child class")

    def print_callback(self, i_batch, metrics, prefix=""):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class Anomaly_Detection_Runner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(Anomaly_Detection_Runner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        try:
            for i, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                X, targets = batch
                X = X.float().to(device=self.device)
                targets = targets.float().to(device=self.device)
                outputs = self.model(X)

                loss = self.loss_module(outputs, targets)
                batch_loss = loss.sum()
                mean_loss = loss.mean()

                backward_loss = mean_loss

                backward_loss.backward()
                self.optimizer.step()

                metrics = {"loss": mean_loss.item()}
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Training " + ending)

                with torch.no_grad():
                    total_samples += loss.numel()
                    epoch_loss += batch_loss.item()  # add total loss of batch

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "outputs": [],
            "metrics": [],
        }

        try:
            for i, batch in enumerate(self.dataloader):

                X, targets = batch
                X = X.float().to(self.device)
                targets = targets.float().to(device=self.device)
                outputs = self.model(X)

                loss = self.loss_module(outputs, targets)
                batch_loss = loss.sum()
                mean_loss = loss.mean()  # mean loss (over samples)
                # (batch_size,) loss for each sample in the batch

                per_batch["targets"].append(targets.half().cpu().numpy())
                per_batch["outputs"].append(outputs.half().cpu().numpy())
                per_batch["metrics"].append([loss.half().cpu().numpy()])

                metrics = {
                    "loss": mean_loss,
                }
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Evaluating " + ending)

                total_samples += loss.numel()
                epoch_loss += batch_loss.half().cpu().item()

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        pred = torch.from_numpy(np.concatenate(per_batch["outputs"], axis=0))
        test_labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)

        # get threshold
        best_threshold, _, _ = eval_mcc(test_labels, pred)
        pred = (pred > best_threshold).cpu().numpy()
        gt = np.array(test_labels).astype(int)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        mcc = matthews_correlation(gt, pred).item()
        auc = roc_auc_score(gt, pred)

        self.epoch_metrics["accuracy"] = accuracy
        self.epoch_metrics["precision"] = precision
        self.epoch_metrics["recall"] = recall
        self.epoch_metrics["f1"] = f_score
        self.epoch_metrics["mcc"] = mcc
        self.epoch_metrics["auc"] = auc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics

    def test(self, epoch_num=None, keep_all=True, threshold=0.5):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "outputs": [],
            "metrics": [],
        }

        try:
            for i, batch in enumerate(self.dataloader):

                X, targets = batch
                X = X.float().to(self.device)
                targets = targets.float().to(device=self.device)
                outputs = self.model(X)

                loss = self.loss_module(outputs, targets)
                batch_loss = loss.sum()
                mean_loss = loss.mean()  # mean loss (over samples)
                # (batch_size,) loss for each sample in the batch

                per_batch["targets"].append(targets.half().cpu().numpy())
                per_batch["outputs"].append(outputs.half().cpu().numpy())
                per_batch["metrics"].append([loss.half().cpu().numpy()])

                metrics = {
                    "loss": mean_loss,
                }
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Evaluating " + ending)

                total_samples += loss.numel()
                epoch_loss += batch_loss.half().cpu().item()

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        pred = torch.from_numpy(np.concatenate(per_batch["outputs"], axis=0))
        test_labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)

        # get threshold
        best_threshold = threshold
        pred = (pred > best_threshold).cpu().numpy()
        gt = np.array(test_labels).astype(int)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        mcc = matthews_correlation(gt, pred).item()
        auc = roc_auc_score(gt, pred)

        self.epoch_metrics["accuracy"] = accuracy
        self.epoch_metrics["precision"] = precision
        self.epoch_metrics["recall"] = recall
        self.epoch_metrics["f1"] = f_score
        self.epoch_metrics["mcc"] = mcc
        self.epoch_metrics["auc"] = auc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class Cluster_Runner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(Cluster_Runner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        try:
            for i, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                X, targets = batch
                X = X.float().to(device=self.device)
                x1 = torch.stack([augment_signal(sig)[0] for sig in X])
                x2 = torch.stack([augment_signal(sig)[0] for sig in X])

                z1 = self.model(x1)
                z2 = self.model(x2)

                loss = self.loss_module(z1, z2)
                batch_loss = loss.sum()
                mean_loss = loss.mean()

                backward_loss = mean_loss

                backward_loss.backward()
                self.optimizer.step()

                metrics = {"loss": mean_loss.item()}
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Training " + ending)

                with torch.no_grad():
                    total_samples += loss.numel()
                    epoch_loss += batch_loss.item()  # add total loss of batch

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "outputs": [],
            "metrics": [],
        }

        try:
            for i, batch in enumerate(self.dataloader):

                X, targets = batch
                X = X.float().to(self.device)
                x1 = torch.stack([augment_signal(sig)[0] for sig in X])
                x2 = torch.stack([augment_signal(sig)[0] for sig in X])

                z1 = self.model(x1)
                z2 = self.model(x2)

                loss = self.loss_module(z1, z2)
                batch_loss = loss.sum()
                mean_loss = loss.mean()  # mean loss (over samples)
                # (batch_size,) loss for each sample in the batch

                per_batch["targets"].append(targets.half().cpu().numpy())
                per_batch["outputs"].append(loss.half().cpu().numpy())  # dummy
                per_batch["metrics"].append([loss.half().cpu().numpy()])

                metrics = {
                    "loss": mean_loss,
                }
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Evaluating " + ending)

                total_samples += loss.numel()
                epoch_loss += batch_loss.half().cpu().item()

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class ClassificationRunner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(ClassificationRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        try:
            for i, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                X, targets = batch
                X = X.float().to(device=self.device)
                targets = targets.long().to(device=self.device)
                outputs = self.model(X)

                loss = self.loss_module(outputs, targets)
                batch_loss = loss.sum()
                mean_loss = loss.mean()

                backward_loss = mean_loss

                backward_loss.backward()
                self.optimizer.step()

                metrics = {"loss": mean_loss.item()}
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Training " + ending)

                with torch.no_grad():
                    total_samples += loss.numel()
                    epoch_loss += batch_loss.item()  # add total loss of batch
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=False):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "outputs": [],
            "metrics": [],
        }

        try:
            for i, batch in enumerate(self.dataloader):

                X, targets = batch
                X = X.float().to(device=self.device)
                targets = targets.long().to(device=self.device)
                outputs = self.model(X)

                loss = self.loss_module(outputs, targets)
                batch_loss = loss.sum()
                mean_loss = loss.mean()  # mean loss (over samples)
                # (batch_size,) loss for each sample in the batch

                per_batch["targets"].append(targets.half().cpu().numpy())
                per_batch["outputs"].append(outputs.half().cpu().numpy())
                per_batch["metrics"].append([loss.half().cpu().numpy()])

                metrics = {
                    "loss": mean_loss,
                }
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                    self.print_callback(i, metrics, prefix="Evaluating " + ending)

                total_samples += loss.numel()
                epoch_loss += batch_loss.half().cpu().item()
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected (Ctrl+C)")
            sys.exit(0)
            del self.dataloader

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        pred = torch.from_numpy(np.concatenate(per_batch["outputs"], axis=0))
        test_labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)

        pred = np.argmax(pred, axis=1)
        gt = np.array(test_labels).astype(int)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        mcc = matthews_correlation(gt, pred).item()

        self.epoch_metrics["accuracy"] = accuracy
        self.epoch_metrics["precision"] = precision
        self.epoch_metrics["recall"] = recall
        self.epoch_metrics["f1"] = f_score
        self.epoch_metrics["mcc"] = mcc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


def validate(
    val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch, fold_i=0
):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info(
        "Validation runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Epoch {} Validation Summary: ".format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar(f"{k}/val_fold_{fold_i}", v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    if config.key_metric in NEG_METRICS:
        condition = aggr_metrics[config.key_metric] < best_value
    else:
        condition = aggr_metrics[config.key_metric] > best_value
    if condition:
        best_value = aggr_metrics[config.key_metric]
        utils.save_model(
            os.path.join(config.save_dir, f"model_best_fold_{fold_i}.pth"),
            epoch,
            val_evaluator.model,
        )
        best_metrics = aggr_metrics.copy()

        if config.task == "cluster":
            return aggr_metrics, best_metrics, best_value

        # save per-batch predictions
        logits = torch.from_numpy(
            np.concatenate(per_batch["outputs"], axis=0)
        )  # [N, 2]
        test_labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)

        df = pd.DataFrame(
            {
                "pred": logits,
                "targets": test_labels,
            }
        )
        df.to_csv(os.path.join(config.pred_dir, f"val_pred_{fold_i}.csv"), index=False)

    return aggr_metrics, best_metrics, best_value


def test(test_evaluator, val_evaluator, config, fold_i=0):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Testing on test set ...")
    with torch.no_grad():
        val_aggr_metrics, val_per_batch = val_evaluator.evaluate(keep_all=True)
        del val_aggr_metrics
    # get best threshold from validation set
    val_logits = torch.from_numpy(
        np.concatenate(val_per_batch["outputs"], axis=0)
    )  # [N, 2]
    val_labels = np.concatenate(val_per_batch["targets"], axis=0).reshape(-1)
    best_threshold, _, _ = eval_mcc(val_labels, val_logits)
    logger.info(f"Fold {fold_i} best test threshold: {best_threshold}")

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = test_evaluator.test(
            keep_all=True, threshold=best_threshold
        )
        del aggr_metrics["epoch"]
    eval_runtime = time.time() - eval_start_time
    logger.info(
        "Testing runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(test_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(test_evaluator.dataloader.dataset)
    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch test. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample test. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Testing Summary: "
    for k, v in aggr_metrics.items():
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    # save per-batch predictions
    logits = torch.from_numpy(np.concatenate(per_batch["outputs"], axis=0))  # [N, 2]
    test_labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)

    df = pd.DataFrame(
        {
            "pred": logits,
            "targets": test_labels,
        }
    )
    df.to_csv(os.path.join(config.pred_dir, f"test_pred_{fold_i}.csv"), index=False)

    return aggr_metrics


def eval_prototype_distance(model):
    with torch.no_grad():
        P = model.proto.prototypes  # [K, D]
        dist = torch.cdist(P, P)  # [K, K]

    print("Prototype distance matrix:")
    print(dist.cpu().numpy())
    print(
        "Mean off-diagonal distance:",
        dist[~torch.eye(dist.size(0), dtype=bool)].mean().item(),
    )

    import math


def eval_assignment_entropy(assign):
    # assign: [B, N, K]
    eps = 1e-8
    entropy = -(assign * (assign + eps).log()).sum(-1)  # [B, N]
    entropy = entropy.mean().item()

    print(f"Mean assignment entropy: {entropy:.4f}")
    print(f"log(K): {math.log(assign.size(-1)):.4f}")


def eval_prototype_usage(assign):
    # assign: [B, N, K]
    usage = assign.mean(dim=(0, 1))  # [K]
    print("Prototype usage:")
    for i, u in enumerate(usage):
        print(f"  Proto {i}: {u.item():.4f}")


def visualize_embedding(z, assign, num_samples=10000):
    """
    z: [B, N, D]
    assign: [B, N, K]
    """
    z = z.reshape(-1, z.size(-1)).cpu().numpy()
    label = assign.argmax(-1).reshape(-1).cpu().numpy()

    idx = np.random.choice(len(z), min(num_samples, len(z)), replace=False)
    z = z[idx]
    label = label[idx]

    tsne = TSNE(n_components=2, perplexity=30, init="pca")
    z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(6, 5))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=label, s=3, cmap="tab10")
    plt.title("Pulse embedding colored by prototype")
    plt.show()


def eval_clustering_metrics(z, assign):
    z = z.reshape(-1, z.size(-1)).cpu().numpy()
    label = assign.argmax(-1).reshape(-1).cpu().numpy()

    sil = silhouette_score(z, label)
    db = davies_bouldin_score(z, label)

    print(f"Silhouette score: {sil:.4f}")
    print(f"Davies-Bouldin index: {db:.4f}")


def evaluate_prototype_model(model, dataloader, device):
    model.eval()

    all_z = []
    all_assign = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, targets = batch
            x = x.to(device)
            z = model.encoder(x)
            stats = model(x)

            all_z.append(z)
            all_assign.append(stats["assign"])

    z = torch.cat(all_z, dim=0)
    assign = torch.cat(all_assign, dim=0)

    eval_prototype_distance(model)
    eval_assignment_entropy(assign)
    eval_prototype_usage(assign)
    visualize_embedding(z, assign)
    eval_clustering_metrics(z, assign)
