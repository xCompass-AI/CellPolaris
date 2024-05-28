import json
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from torch.utils.data import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from load_dataset import PECADataset, RNASeqStatisticsFeature


@torch.no_grad()
def build_pseudo_loader(
    model,
    rna_seqs: Iterable,
    rna_seq_masks: Iterable,
    tf_mask,
    tg_mask,
    num_samples,
    batch_size,
):
    r"""
    Build pseudo target train loader for training.

    Args: similar to generate_grn function.
        rna_seqs: rna_seq of each domain.
        rna_seq_masks: It indicates which rna_seq is useful.
        tf_mask:
        tg_mask:
        num_samples: number of samples to preserve.
        batch_size: batch size of each pseudo loader.

    Returns:
        pseudo_pos_loaders: list of pseudo positive loaders
        pseudo_neg_loaders: list of pseudo negative loaders
    """
    pseudo_pos_loaders, pseudo_neg_loaders = [], []
    for rna_seq, rna_seq_mask in zip(rna_seqs, rna_seq_masks):
        edge_index, edge_weight = generate_grn(
            model, rna_seq, rna_seq_mask, tf_mask, tg_mask, enable_tqdm=False
        )
        pos_edge_index, pos_edge_weight = filter_grn_by_topk(
            edge_index, edge_weight, num_samples
        )


        candidate_neg_edge_index, candidate_neg_edge_weight = filter_grn_by_topk(
            edge_index, -edge_weight, num_samples * 20
        )
        perm = torch.randperm(candidate_neg_edge_index.shape[1])[:num_samples]
        neg_edge_index = candidate_neg_edge_index[:, perm]
        neg_edge_weight = candidate_neg_edge_weight[perm]

        pseudo_pos_dataset = PECADataset(
            pos_edge_index.detach().cpu(),
            pos_edge_weight.detach().cpu(),
            rna_seq.detach().cpu(),
            rna_seq_mask.detach().cpu(),
        )
        pseudo_pos_loader = DataLoader(
            pseudo_pos_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        pseudo_pos_loaders.append(pseudo_pos_loader)

        pseudo_neg_dataset = PECADataset(
            neg_edge_index.detach().cpu(),
            neg_edge_weight.detach().cpu(),
            rna_seq.detach().cpu(),
            rna_seq_mask.detach().cpu(),
        )
        pseudo_neg_loader = DataLoader(
            pseudo_neg_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        pseudo_neg_loaders.append(pseudo_neg_loader)

    return pseudo_pos_loaders, pseudo_neg_loaders


@torch.no_grad()
def generate_grn(model, rna_seq, rna_seq_mask, tf_mask, tg_mask, enable_tqdm=True):
    r"""
    Args:
        rna_seq: rna_seq
        rna_seq_mask: It indicates which rna_seq is existed in this peca.
        tf_mask: It indicates which genes are tf.
        tg_mask: It indicates which genes are tg.
        enable_tqdm: whether to enable tqdm.
    """
    num_nodes = rna_seq.shape[0]
    indices = torch.arange(num_nodes, device=rna_seq.device)
    indices[~rna_seq_mask] = -1
    tf_rna_seq = rna_seq[tf_mask]
    tf_indices = indices[tf_mask]
    tg_rna_seq = rna_seq[tg_mask]
    tg_indices = indices[tg_mask]

    x0 = tf_rna_seq.repeat(tg_rna_seq.shape[0], 1)
    x0_indices = tf_indices.repeat(tg_rna_seq.shape[0])
    x1 = tg_rna_seq.repeat_interleave(tf_rna_seq.shape[0], dim=0)
    x1_indices = tg_indices.repeat_interleave(tf_rna_seq.shape[0])
    edge_index = torch.stack([x0_indices, x1_indices], dim=0)

    mask = torch.logical_and(x0_indices != -1, x1_indices != -1)
    x0 = x0[mask]
    x1 = x1[mask]
    edge_index = edge_index[:, mask]

    batch_size = 10000 * 100
    edge_weight = torch.zeros(edge_index.shape[1], device=x0.device)
    if enable_tqdm:
        tqdm_bar = tqdm(total=edge_weight.shape[0] // batch_size + 1)

    for index, (x0_, x1_, edges_) in enumerate(
        zip(
            torch.split(x0, batch_size),
            torch.split(x1, batch_size),
            torch.split(edge_index, batch_size, dim=1),
        )
    ):
        edge_weight[index * batch_size : (index + 1) * batch_size] = model(
            x0_, x1_, edges_
        )[0]
        if enable_tqdm:
            tqdm_bar.update()
    return edge_index, edge_weight


def filter_grn_by_threshold(edge_index, edge_weight, threshold=0.05):
    mask = edge_weight >= threshold
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]
    return edge_index, edge_weight


def filter_grn_by_ratio(edge_index, edge_weight, ratio=0.1):
    return filter_grn_by_topk(
        edge_index, edge_weight, int(edge_weight.shape[0] * ratio)
    )


def filter_grn_by_topk(edge_index, edge_weight, k=1000):
    k = min(int(k), edge_weight.shape[0])
    indices = torch.topk(edge_weight, int(k), dim=0)[1]
    edge_index = edge_index[:, indices]
    edge_weight = edge_weight[indices]
    return edge_index, edge_weight


def evaluate_generated_grn(
    pred_edge_index, pred_edge_weight, true_edge_index, true_edge_weight, save_path
):
    pred_edge_index, pred_edge_weight = remove_self_loops(
        pred_edge_index, pred_edge_weight
    )
    true_edge_index, true_edge_weight = remove_self_loops(
        true_edge_index, true_edge_weight
    )
    pred_edge_index, pred_edge_weight = filter_grn_by_threshold(
        pred_edge_index, pred_edge_weight, 0.04
    )
    pred_grns = {}
    true_grns = {}
    select_ratios = [0.25, 0.5, 0.75, 1]
    times = [1, 2, 3]
    for i in times:
        for j in select_ratios:
            num_true = true_edge_weight.shape[0] * j
            pred_grns[i * j] = filter_grn_by_topk(
                pred_edge_index, pred_edge_weight, int(num_true * i)
            )
    for i in select_ratios:
        true_grns[i] = filter_grn_by_ratio(true_edge_index, true_edge_weight, i)

    results = defaultdict(dict)
    for i in times:
        for j in select_ratios:
            print(f"pred: {i}, true: {j}")
            results[i][j] = evaluate_similarity_of_two_graphs(
                pred_grns[i * j][0], true_grns[j][0]
            )
    draw_venn_graph(results, save_path.with_suffix(save_path.suffix + ".venn.png"))
    with open(save_path.with_suffix(save_path.suffix + ".venn.json"), "w") as fp:
        result = {
            f"pred_{i}x_true_{j}": x[1]
            for i, data in results.items()
            for j, x in data.items()
        }
        json.dump(result, fp, indent=4)
    return results


def evaluate_similarity_of_two_graphs(pred_edge_index, true_edge_index):
    r"""
    They have the same vertices.

    Return:
        similarity
        recall
        precision
    """
    true_edge_index = remove_self_loops(true_edge_index)[0]
    pred_edge_index = pred_edge_index.cpu().t().tolist()
    true_edge_index = true_edge_index.cpu().t().tolist()
    pred_edge_index = set(map(tuple, pred_edge_index))
    true_edge_index = set(map(tuple, true_edge_index))

    pred_in_true = len(pred_edge_index & true_edge_index)
    pred_not_in_true = len(pred_edge_index - true_edge_index)
    true_not_in_pred = len(true_edge_index - pred_edge_index)

    similarity = pred_in_true / (pred_in_true + pred_not_in_true + true_not_in_pred)
    recall = pred_in_true / (pred_in_true + true_not_in_pred)
    precision = pred_in_true / (pred_in_true + pred_not_in_true)
    return (
        [pred_in_true, pred_not_in_true, true_not_in_pred],
        {"similarity": similarity, "recall": recall, "precision": precision},
    )


def draw_venn_graph(results, path):
    nrows = results.keys().__len__()
    ncols = results[results.keys().__iter__().__next__()].keys().__len__()
    fig = plt.figure(figsize=(4.5 * ncols, 4 * nrows))
    subfigs = fig.subfigures(nrows=nrows, ncols=ncols, squeeze=False)
    for idx1, (i, result) in enumerate(results.items()):
        for idx2, (j, data) in enumerate(result.items()):
            ax = subfigs[idx1][idx2].subplots()
            pred_in_true, pred_not_in_true, true_not_in_pred = data[0]

            venn2(
                subsets=(
                    pred_not_in_true,
                    true_not_in_pred,
                    pred_in_true,
                ),
                set_labels=("Predicted", "True"),
                set_colors=("blue", "green"),
                alpha=0.5,
                ax=ax,
            )

            subfigs[idx1][idx2].suptitle(
                f"pred: {i:.1f}x, true: {j*100:.1f}%\n recall: {data[1]['recall']:.3f}, prec: {data[1]['precision']:.3f}"
            )

            subfigs[idx1][idx2].set_facecolor(
                (
                    1 - data[1]["recall"] / 3,
                    1 - data[1]["recall"] / 3,
                    1 - data[1]["recall"] / 3,
                )
            )

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def process_externel_rna_seq(
    rna_seq: pd.DataFrame,
    gene_names,
    preserve_gene_mask,
    is_bulk=False,
    is_sc=False,
    is_mouse=False,
    is_human=False,
    mapping=None,
):
    r"""
    Args:
        rna_seq: pd.DataFrame, index is gene_name
        gene_names: (from dataset) genes which we want to preserve
        preserve_gene_mask: (from dataset) genes which we can handle
        is_bulk:
        is_sc:
        is_mouse:
        is_human:
        mapping: if provide, it means to mapping convert the raw gene_names
    """
    if is_human:
        if mapping is not None:
            rna_seq.index = rna_seq.index.map(
                lambda x: mapping[x] if x in mapping else x
            )
        else:
            raise ValueError("mapping must be provided for human data")

    out_rna_seq = np.zeros(len(gene_names)) - 1
    for index, (gene_name, is_preserve) in enumerate(
        zip(gene_names, preserve_gene_mask)
    ):
        if is_preserve and gene_name in rna_seq.index:
            out_rna_seq[index] = rna_seq.loc[gene_name].iloc[0]
    out_rna_seq = torch.FloatTensor(out_rna_seq).t().squeeze()
    mask = out_rna_seq >= 0
    print(mask.sum().item())
    out_rna_seq = RNASeqStatisticsFeature()(
        out_rna_seq, mask, is_bulk, is_sc, is_mouse, is_human
    )
    return out_rna_seq, torch.BoolTensor(mask)


def dump_generated_grn(
    edge_index,
    edge_weight,
    save_to_path,
    gene_names,
    out_is_human=False,
    human_to_mouse_mapping=None,
    cross_species=False,
):
    r"""
    out_is_human : if provide, it means to mapping convert the raw gene_names.
        We will convert the gene_names to `human' gene_names.
    human_to_mouse_mapping : Homeotic genes mapping.
    cross_species: if set to True, we only keep homeotic genes. Note that,
        human_to_mouse_mapping should be provided.
    """
    homeotic_genes = set(human_to_mouse_mapping.values())

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = filter_grn_by_ratio(edge_index, edge_weight, ratio=0.1)
    if out_is_human:
        mapping = {v: k for k, v in human_to_mouse_mapping.items()}

    id_to_gene = {i: gene for i, gene in enumerate(gene_names)}
    edge_index = edge_index.cpu().t().tolist()
    edge_weight = edge_weight.cpu().tolist()
    with open(save_to_path, "w") as fp:
        fp.write("TF\tTG\tScore\n")
        for (u, v), w in zip(edge_index, edge_weight):
            u = id_to_gene[u]
            v = id_to_gene[v]
            if cross_species:
                if u not in homeotic_genes or v not in homeotic_genes:
                    print("ignore", u, v)
                    continue
            if out_is_human:
                u = mapping[u] if u in mapping else u
                v = mapping[v] if v in mapping else v
            fp.write(f"{u}\t{v}\t{w}\n")
