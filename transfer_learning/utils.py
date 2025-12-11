from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import chain
from multiprocessing.pool import ThreadPool
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric import seed_everything


class Metrics:
    @staticmethod
    def calc_smape(A, F):
        return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

    def calc_metric(self, pred, target, threshold=0.1, ignore_binary_metric=False):
        r"""
        Args:
            pred: [N, 1]
            target: [N, 1]
            threshold: threshold for positive edges, edges larger than the threshold are
                considered as positive edges. Set to None to consider all edges as positive.
            ignore_binary_metric: whether to ignore the binary classification metrics.
        """

        if threshold is None:
            threshold = 0.0
        pred = deepcopy(pred)
        target = deepcopy(target)

        regression_pred = pred[target >= threshold]
        regression_target = target[target >= threshold]
        r2 = r2_score(regression_target, regression_pred)
        mape = mean_absolute_percentage_error(
            regression_target + 1, regression_pred + 1
        )
        smape = self.calc_smape(regression_target + 1, regression_pred + 1)

        mse = ((regression_pred - regression_target) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(regression_target, regression_pred)

        if not ignore_binary_metric:
            target[target >= threshold] = True
            target[target < threshold] = False
            pred[pred >= threshold] = True
            pred[pred < threshold] = False
            acc = (target == pred).sum() / len(target)
            recall = (target * pred).sum() / target.sum()
            precision = (target * pred).sum() / pred.sum()
            f1 = 2 * recall * precision / (recall + precision)
            roc_auc = roc_auc_score(target, pred)
            average_precision = average_precision_score(target, pred)
        else:
            acc = recall = precision = f1 = roc_auc = average_precision = -1

        return {
            "r2": float(r2),
            "mape": float(mape),
            "smape": float(smape),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "acc": float(acc),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "auprc": float(average_precision),
        }

    def metric_format(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold=None,
        ignore_binary_metric=False,
    ):
        r"""
        Args:
            pred: [N, 1]
            target: [N, 1]
            threshold: threshold for positive value.
        """
        metric = self.calc_metric(pred, target, threshold, ignore_binary_metric)
        return (
            f"r2: {metric['r2']:.4f},"
            f"MAPE: {metric['mape']:.4f}, "
            f"RMSE: {metric['rmse']:.4f}, "
            f"MAE: {metric['mae']:.4f}, "
            f"Acc: {metric['acc']:.4f}, "
            f"Recall: {metric['recall']:.4f}, "
            f"Prec: {metric['precision']:.4f}, "
            f"F1: {metric['f1']:.4f}, "
            f"roc_auc: {metric['roc_auc']:.4f}, "
            f"AUPRC: {metric['auprc']:.4f}"
        )


def infinite_loader_with_domain_index(loader, index):
    while True:
        for batch in loader:
            yield *batch, torch.LongTensor([index] * len(batch[0]))


def get_data_from_loaders(loaders: Iterable, device):
    r"""
    Get data from loaders, and move to device

    Args:
        loaders: iterable of loaders
        device: torch.device

    Returns:
        x0_list: list of x0 of each domain
        x1_list: list of x1 ...
        edges_list: list of edges ...
        y_list: list of y ...
        tissue_index_list: list of tissue_index ...
    """
    with ThreadPool(6) as pool:
        data = pool.map(next, loaders)
    x0_list, x1_list, edges_list, y_list, tissue_index_list = zip(*data)

    lengths = [len(x0) for x0 in x0_list]
    x0_list = torch.split(torch.cat(x0_list, dim=0).to(device), lengths)
    x1_list = torch.split(torch.cat(x1_list, dim=0).to(device), lengths)
    edges_list = torch.split(torch.cat(edges_list, dim=0).to(device), lengths)
    y_list = torch.split(torch.cat(y_list, dim=0).to(device), lengths)
    tissue_index_list = torch.split(
        torch.cat(tissue_index_list, dim=0).to(device), lengths
    )
    return x0_list, x1_list, edges_list, y_list, tissue_index_list


def _get_data(loader):
    x0, x1, edges, y = [], [], [], []
    for batch in loader:
        x0.append(batch[0])
        x1.append(batch[1])
        edges.append(batch[2])
        y.append(batch[3])

    return x0, x1, edges, y


def build_chain_loader(loaders, batch_size=2560):
    r"""
    Merge each loaders into one loader to accelerate validation and testing.
    Also increasing the batch size.
    """
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = pool.map(_get_data, loaders)
    x0_list, x1_list, edges_list, y_list = zip(*results)

    dataset = TensorDataset(
        torch.vstack(list(chain.from_iterable(x0_list))),
        torch.vstack(list(chain.from_iterable(x1_list))),
        torch.vstack(list(chain.from_iterable(edges_list))),
        torch.hstack(list(chain.from_iterable(y_list))),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
    )
    return loader


def build_filtered_chain_loader(loaders, batch_size=2560):
    r"""
    Merge each loaders into one loader to accelerate validation and testing.
    Also increasing the batch size.
    """
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = pool.map(_get_data, loaders)
    x0_list, x1_list, edges_list, y_list = zip(*results)

    x0_list, x1_list, edges_list, y_list = _get_filtered_data(x0_list, x1_list, edges_list, y_list)

    dataset = TensorDataset(
        x0_list,
        x1_list,
        edges_list,
        y_list,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
    )

    return loader

def _get_filtered_data(x0_list, x1_list, edges_list, y_list):
    x0_list = torch.vstack(list(chain.from_iterable(x0_list))),
    x1_list = torch.vstack(list(chain.from_iterable(x1_list))),
    edges_list = torch.vstack(list(chain.from_iterable(edges_list))),
    edges_list = edges_list[0]
    y_list = torch.hstack(list(chain.from_iterable(y_list))),
    y_list= y_list[0]
    valid_indices = np.where(y_list > 0.1)[0]
    non_equal_indices = np.where(edges_list[:, 0] != edges_list[:, 1])[0]
    unique_rows, unique_indices = np.unique(edges_list, axis=0, return_index=True)
    selected_indices = np.intersect1d(valid_indices, non_equal_indices)
    selected_indices = np.intersect1d(selected_indices, unique_indices)

    selected_x0_list = x0_list[0][selected_indices]
    selected_x1_list = x1_list[0][selected_indices]
    selected_edges_list = edges_list[selected_indices]
    selected_y_list = y_list[selected_indices]

    return selected_x0_list, selected_x1_list, selected_edges_list, selected_y_list


def draw_histogram(delta_y, path):
    hist = np.histogram(delta_y, bins=100, density=True)
    plt.bar(hist[1][:-1], hist[0], width=hist[1][1] - hist[1][0], edgecolor="black")
    x = np.linspace(np.min(delta_y), np.max(delta_y), 100)
    mu, sigma = np.mean(delta_y), np.std(delta_y)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
    plt.plot(x, pdf, color="red", linewidth=2)

    plt.xlabel("Predicted Edge Weight - True Edge Weight")
    plt.ylabel("Count (in %)")
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()


def get_cross_validation_mask(num: int, seed: int, fold=4):
    assert fold > 1
    seed_everything(seed)
    mask = torch.randperm(num)
    num_each_fold = num // fold
    mask = list(torch.split(mask, num_each_fold))
    if len(mask) > fold:
        extra_domains = torch.cat(mask[fold:])
        mask = mask[:fold]
        for i in range(len(extra_domains)):
            mask[i % fold] = torch.cat([mask[i % fold], extra_domains[i].unsqueeze(0)])
    return mask


def get_exact_peca_mask():
    return 0