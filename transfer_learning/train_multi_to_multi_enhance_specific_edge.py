import argparse
import json
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

import torch
torch.backends.cuda.matmul.allow_tf32 = True

from torch_geometric import seed_everything
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from load_dataset import GRNPredictionDataset
from models import TRModel, build_model
from utils import (
    Metrics,
    draw_histogram,
    get_cross_validation_mask,
    infinite_loader_with_domain_index,
)
import warnings
from multiprocessing.pool import ThreadPool
from typing import Iterable
import copy

warnings.filterwarnings("error")

def get_species(species):
    name = []
    if len(species)==1:
        name = species[0]
    else:
        for i in species: 
            if len(name) == 0:
                name = i
            else:
                name = name + '_' + i
    return name

def get_data_from_loaders_addspecificmask(loaders: Iterable, device):
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
    x0_list, x1_list, edges_list, y_list, spec_mask_list, tissue_index_list = zip(*data)

    lengths = [len(x0) for x0 in x0_list]
    x0_list = torch.split(torch.cat(x0_list, dim=0).to(device), lengths)
    x1_list = torch.split(torch.cat(x1_list, dim=0).to(device), lengths)
    edges_list = torch.split(torch.cat(edges_list, dim=0).to(device), lengths)
    y_list = torch.split(torch.cat(y_list, dim=0).to(device), lengths)
    spec_mask_list = torch.split(torch.cat(spec_mask_list, dim=0).to(device), lengths)
    tissue_index_list = torch.split(
        torch.cat(tissue_index_list, dim=0).to(device), lengths
    )
    return x0_list, x1_list, edges_list, y_list, spec_mask_list, tissue_index_list


def train(loader_iters, n_iter=10, dg_loss_weight=1, device="cpu"):
    model.train()

    total_loss = 0
    y_pred, y_true = [], []
    for _ in range(n_iter):
        x0_list, x1_list, edges_list, y_list, spec_mask_list, tissue_index_list = get_data_from_loaders_addspecificmask(
            loader_iters, device
        )
        optimizer.zero_grad()
        out, transfer_loss = model(
            x0_list, x1_list, edges_list, tissue_index_list, y_list
        )
        y = torch.cat(y_list, dim=0)

        if not config.ignore_regression:
            regression_loss = criterion(out, y)
            regression_loss = torch.mean(regression_loss)
        else:
            regression_loss = 0

        spec_mask_list = torch.cat(spec_mask_list, dim=0)
        regression_loss_spec = torch.where(spec_mask_list, regression_loss * 100, regression_loss)
        regression_loss_spec = torch.mean(regression_loss_spec)

        if transfer_loss is not None:
            if config.dg_loss_type == "graph_mixup":
                loss = regression_loss_spec + dg_loss_weight * transfer_loss
            else:
                loss = regression_loss + dg_loss_weight * transfer_loss
        else:
            loss = regression_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(y)
        y_pred.append(out.cpu().detach())
        y_true.append(y.cpu().detach())
    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    return total_loss / y_pred.shape[0], y_pred, y_true

@torch.no_grad()
def test(loader):
    model.eval()

    total_loss = 0
    y_pred, y_true, spec_mask_all = [], [], []
    for x0, x1, edges, y, spec_mask in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        edges = edges.t().to(device)
        y = y.to(device)
        spec_mask = spec_mask.to(device)

        out, _ = model.link_prediction_model(x0, x1, edges)
        loss = criterion(out, y)
        # mean
        loss = torch.mean(loss)
        total_loss += float(loss) * len(y)
        y_pred.append(out.cpu().detach())
        y_true.append(y.cpu().detach())
        spec_mask_all.append(spec_mask.cpu().detach())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    spec_mask_all = torch.cat(spec_mask_all, dim=0).numpy()
    return total_loss / y_pred.shape[0], y_pred, y_true, spec_mask_all


def get_common_process_loaders(train_loaders, test_loaders):
    print("Loading all dataset...")
    all_train_loaders = {}
    all_test_loaders = {}
    if "bulk_mouse" in train_loaders.keys():
        all_train_loaders["bulk_mouse"], all_test_loaders["bulk_mouse"] = [], []
        for tissue_name, tissue in train_loaders["bulk_mouse"].items():
            for period_name, loader in tissue.items():
                all_train_loaders["bulk_mouse"].append(loader)
                all_test_loaders["bulk_mouse"].append(
                    test_loaders["bulk_mouse"][tissue_name][period_name]
                )
    for species in ["bulk_human", "sc_mouse", "sc_human"]:
        if species in train_loaders.keys():
            all_train_loaders[species], all_test_loaders[species] = [], []
            for name, loader in train_loaders[species].items():
                all_train_loaders[species].append(loader)
                all_test_loaders[species].append(test_loaders[species][name])

    for name, loader in train_loaders["bulk_mouse_master_tf"].items():
        all_train_loaders["bulk_mouse"].append(loader)
        all_test_loaders["bulk_mouse"].append(
            test_loaders["bulk_mouse_master_tf"][name]
        )
    print("Done.")

    print("Getting required loaders...")
    source_train_loaders, source_test_loaders = [], []
    target_train_loaders, target_test_loaders = [], []

    species = config.species
    if all(x in ["bulk_mouse", "bulk_human", "sc_mouse", "sc_human"] for x in species):
        if len(species)>1:
            for spe in species:
                train_loaders_tmp, test_loaders_tmp = [], []
                if (spe == "sc_human") or (spe=="bulk_human"):
                    if spe == "sc_human" and config.append_bulk:
                        bulk_train_loaders = all_train_loaders["bulk_human"]
                        bulk_test_loaders = all_test_loaders["bulk_human"]

                    train_loaders_tmp = all_train_loaders[spe]
                    test_loaders_tmp = all_test_loaders[spe]
                    mask = get_cross_validation_mask(len(train_loaders_tmp), config.seed, fold=4)
                    train_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()
                    test_mask = mask[config.fold].tolist()
                    for index in train_mask:
                        source_train_loaders.append(train_loaders_tmp[index])
                        source_test_loaders.append(test_loaders_tmp[index])
                    for index in test_mask:
                        target_train_loaders.append(train_loaders_tmp[index])
                        target_test_loaders.append(test_loaders_tmp[index])
                    if config.append_bulk:
                        source_train_loaders += bulk_train_loaders
                        source_test_loaders += bulk_test_loaders
                elif (spe == "sc_mouse") or (spe=="bulk_mouse"):
                    if spe == "sc_mouse" and config.append_bulk:
                        bulk_train_loaders = all_train_loaders["bulk_mouse"]
                        bulk_test_loaders = all_test_loaders["bulk_mouse"]      
                    train_loaders_tmp = all_train_loaders[spe]
                    mask = get_cross_validation_mask(len(train_loaders_tmp), config.seed, fold=4)
                    train_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()
                    for index in train_mask:
                        source_train_loaders.append(train_loaders_tmp[index])
                    if config.append_bulk:
                        source_train_loaders += bulk_train_loaders
                        source_test_loaders += bulk_test_loaders
            del (
                dataset.datasets,
                all_train_loaders,
                all_test_loaders,
                train_loaders,
                test_loaders,
                train_loaders_tmp,
                test_loaders_tmp,
            )
                    
        else:
            species = species[0]
            if species == "sc_mouse" and config.append_bulk:
                bulk_train_loaders = all_train_loaders["bulk_mouse"]
                bulk_test_loaders = all_test_loaders["bulk_mouse"]
            elif species == "sc_human" and config.append_bulk:
                bulk_train_loaders = all_train_loaders["bulk_human"]
                bulk_test_loaders = all_test_loaders["bulk_human"]

            all_train_loaders = all_train_loaders[species]
            all_test_loaders = all_test_loaders[species]
            mask = get_cross_validation_mask(len(all_train_loaders), config.seed, fold=4)
            train_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()
            test_mask = mask[config.fold].tolist()

            for index in train_mask:
                source_train_loaders.append(all_train_loaders[index])
                source_test_loaders.append(all_test_loaders[index])
            for index in test_mask:
                target_train_loaders.append(all_train_loaders[index])
                target_test_loaders.append(all_test_loaders[index])
            if config.append_bulk:
                source_train_loaders += bulk_train_loaders
                source_test_loaders += bulk_test_loaders

            if species == "sc_mouse" and config.fold == 0:
                source_train_loaders += train_loaders["sc_mouse_hsc"].values()
                source_train_loaders += test_loaders["sc_mouse_placenta"].values()
                source_test_loaders += test_loaders["sc_mouse_hsc"].values()
                source_test_loaders += test_loaders["sc_mouse_placenta"].values()
            del (
                dataset.datasets,
                all_train_loaders,
                all_test_loaders,
                train_loaders,
                test_loaders,
            )
    else:
        raise ValueError(f"Unknown species: {config.species}")
    print("Done.")


    if config.num_source is not None:
        print(f"Tuncate source domain to {config.num_source}.")
        source_train_loaders = source_train_loaders[: config.num_source]

    return source_train_loaders, source_test_loaders, target_train_loaders, target_test_loaders

class PECADataset_spec:
    def __init__(self, x0, x1, edge_index, edge_weight, spe_mask):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.x0 = x0
        self.x1 = x1
        self.len = edge_index.shape[0]
        self.spe_mask = spe_mask

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.x0[index],
            self.x1[index],
            self.edge_index[index],
            self.edge_weight[index],
            self.spe_mask[index],
        )


def get_specific_process_dataloader(loaders, threshold=0.1):
    x0, x1, edges, y = [], [], [], []
    len_each_loader = torch.zeros(len(loaders), dtype=torch.int)
    for i in range(len(loaders)):
        loaders = copy.deepcopy(loaders)
        if loaders[i].drop_last == True:
            if_train = True
            loaders[i] = DataLoader(
                dataset=loaders[i].dataset,
                batch_size=loaders[i].batch_size,
                shuffle=False,
                drop_last=False
                )
        else:
            if_train = False
        len_each_loader[i] = len(loaders[i].dataset)
        for batch in loaders[i]:
            x0.append(batch[0])
            x1.append(batch[1])
            edges.append(batch[2])
            y.append(batch[3].reshape(-1,1))

    x0_all = torch.vstack(x0)
    x1_all = torch.vstack(x1)
    edges_all = torch.vstack(edges)
    y_all = torch.vstack(y)
    y_all = y_all.reshape(-1)

    mask = (x0_all[:, 0]>0) & (x0_all[:, 2] == 1)

    pos_edge = torch.zeros(len(edges_all), dtype=torch.bool)
    pos_edge = (y_all>threshold) & (edges_all[:,0]!=edges_all[:,1])
    mask = mask & pos_edge

    edge_counts = {}
    for edge in edges_all[mask].tolist():
        edge_counts[tuple(edge)] = edge_counts.get(tuple(edge), 0) + 1
    threshold_count = int(mask.sum() * 0.05)    
    min_occurrences = sorted(edge_counts.values())[threshold_count]

    indices = []
    for i, edge in enumerate(edges_all[mask].tolist()):
        if edge_counts[tuple(edge)] <= min_occurrences:
            indices.append(i)
    indices = np.array(indices, dtype=int)
    mask_true = np.where(mask==True)[0]
    mask_true = mask_true[indices]
    specific_edge_mask = torch.zeros(len(edges_all), dtype=torch.bool)
    specific_edge_mask[mask_true] = True

    loaders_new = []
    for i in range(len(len_each_loader)):
        if i == 0:
            x0 = x0_all[:len_each_loader[i]]
            x1 = x1_all[:len_each_loader[i]]
            edges = edges_all[:len_each_loader[i]]
            y = y_all[:len_each_loader[i]]
            spe_mask = specific_edge_mask[:len_each_loader[i]]
        else:
            x0 = x0_all[len_each_loader[:i].sum() : (len_each_loader[:i].sum()+len_each_loader[i])]
            x1 = x1_all[len_each_loader[:i].sum() : (len_each_loader[:i].sum()+len_each_loader[i])]
            edges = edges_all[len_each_loader[:i].sum() : (len_each_loader[:i].sum()+len_each_loader[i])]
            y = y_all[len_each_loader[:i].sum() : (len_each_loader[:i].sum()+len_each_loader[i])]
            spe_mask = specific_edge_mask[len_each_loader[:i].sum() : (len_each_loader[:i].sum()+len_each_loader[i])]
        dataset = PECADataset_spec(
            x0,
            x1,
            edges,
            y,
            spe_mask,
        )
        if if_train == True:
            spe_loader = DataLoader(
                        dataset,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=1,
                        drop_last=True,
                    )
        else:
            spe_loader = DataLoader(
                        dataset,
                        batch_size=3000,
                        shuffle=False,
                        num_workers=1,
                    )
        loaders_new.append(spe_loader)
    return loaders_new

def _get_data_spec(loader):
    x0, x1, edges, y, spec_mask = [], [], [], [], []
    for batch in loader:
        x0.append(batch[0])
        x1.append(batch[1])
        edges.append(batch[2])
        y.append(batch[3])
        spec_mask.append(batch[4])

    return x0, x1, edges, y, spec_mask

def build_chain_loader_addspecific(loaders, batch_size=2560):
    r"""
    Merge each loaders into one loader to accelerate validation and testing.
    Also increasing the batch size.
    """
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = pool.map(_get_data_spec, loaders)
    x0_list, x1_list, edges_list, y_list, spec_mask_list = zip(*results)

    dataset = TensorDataset(
        torch.vstack(list(chain.from_iterable(x0_list))),
        torch.vstack(list(chain.from_iterable(x1_list))),
        torch.vstack(list(chain.from_iterable(edges_list))),
        torch.hstack(list(chain.from_iterable(y_list))),
        torch.hstack(list(chain.from_iterable(spec_mask_list))),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
    )
    return loader


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--species", nargs="+", default=["sc_human"])
    args.add_argument(
        "--append_bulk",
        action="store_true",
        default=False,
        help="Append bulk data to the training data.",
    )
    args.add_argument("--fold", "--k-of-4-fold", type=int, default=0)
    args.add_argument(
        "--dg_loss_type",
        type=lambda x: None if x == "None" else x,
        default="graph_mixup",
    )
    args.add_argument("--dg_loss_weight", type=float, default=1)
    args.add_argument("--ignore_regression", action="store_true", default=False)
    args.add_argument("--mixup_alpha", type=float, default=0.2)
    args.add_argument("--top_ratio", type=float, default=0.11, help="if dg_loss_type is graph-based, this is a hype-parameter")
    args.add_argument("--num_source", type=int, default=None)
    args.add_argument("--model", type=str, default="ncf")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--batch_size", type=int, default=1280)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--weight_decay", type=float, default=0.01)
    args.add_argument("--epoch", type=int, default=1)
    args.add_argument("--device", type=str, default="cuda:2")
    args.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test the performance of model.",
    )
    config = args.parse_args()
    if config.dg_loss_type == "None":
        config.dg_loss_type = None
    if "sc" not in config.species:
        config.append_bulk = False
        print("Append bulk data to the training data: False")
    if config.dg_loss_type is None or "mixup" not in config.dg_loss_type:
        config.ignore_regression = False
        print("Ignore regression loss: False")

    dataset = GRNPredictionDataset()
    train_loaders, test_loaders = dataset.get_loaders(config.batch_size)
    source_train_loaders, source_test_loaders, target_train_loaders, target_test_loaders = get_common_process_loaders(train_loaders, test_loaders)
    source_min_n_iter = min(len(x) for x in source_train_loaders)
    source_train_loaders = get_specific_process_dataloader(source_train_loaders, threshold=0.1)
    source_test_loaders = get_specific_process_dataloader(source_test_loaders, threshold=0.1)
    target_test_loaders = get_specific_process_dataloader(target_test_loaders, threshold=0.1)

    source_train_loaders = [
        infinite_loader_with_domain_index(x, index)
        for index, x in enumerate(source_train_loaders)
    ]

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss(reduction='none')
    link_prediction_model = build_model(config.model, dataset.num_nodes)
    model = TRModel(
        link_prediction_model,
        config.dg_loss_type,
        num_sourcedomains=len(source_train_loaders),
        mixup_alpha=config.mixup_alpha,
        ignore_regression=config.ignore_regression,
        top_ratio = config.top_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epoch
    )

    species_name = get_species(config.species)
    base = Path(__file__).parent / "result/multi_to_multi_generalization"
    if config.dg_loss_type == 'graph_mixup':
        file_name = (
            base / f"{config.model}/{config.dg_loss_type}/"
            f"{species_name}/mixupalpha{config.mixup_alpha}_topratio{config.top_ratio}/fold_{config.fold}_{config.epoch}epoch"
        )
    else:
        file_name = (
            base / f"{config.model}/{config.dg_loss_type}/"
            f"{config.species}/fold_{config.fold}"
        )

    if config.append_bulk:
        file_name = (
            base / f"{config.model}/{config.dg_loss_type}/"
            f"{config.species}_with_bulk/fold_{config.fold}"
        )

    if config.num_source is not None:
        file_name = (
            file_name.parent / f"saturation/fold_{config.fold}_{config.num_source}"
        )
    file_name.parent.mkdir(exist_ok=True, parents=True)

    if config.test:
        print("Test mode ...")
        config.epoch = 0
        model.load_state_dict(
            torch.load(
                file_name.with_suffix(file_name.suffix + ".pt"),
                map_location=device,
            )
        )

    best_source_test_loss = float("inf")
    best_source_test_pred, best_source_test_true = None, None
    best_target_test_pred, best_target_test_true = None, None
    best_target_delta_y = None
    best_target_test_specmask = None

    print("Building chain loaders for accelerating ...")
    source_test_loader = build_chain_loader_addspecific(
        source_test_loaders, batch_size=config.batch_size
    )

    target_test_loader = build_chain_loader_addspecific(
        target_test_loaders, batch_size=config.batch_size
    )


    print("Done.")

    print("Start training ...")
    for epoch in tqdm(range(1, config.epoch + 1)):

        train_loss, train_pred, train_true = train(
            source_train_loaders,
            source_min_n_iter,
            config.dg_loss_weight,
            config.device,
        )

        source_test_loss, source_test_pred, source_test_true, _ = test(source_test_loader)

        target_test_loss, target_test_pred, target_test_true, target_test_spec_mask = test(target_test_loader)
        scheduler.step()

        if source_test_loss < best_source_test_loss:
            best_source_test_loss = source_test_loss

            best_target_delta_y = (target_test_pred - target_test_true)[target_test_true > 0.1]
            best_target_test_pred, best_target_test_true, best_target_test_specmask = target_test_pred, target_test_true, target_test_spec_mask
            best_source_test_pred, best_source_test_true = source_test_pred, source_test_true

            print("begin save")
            print(file_name.with_suffix(file_name.suffix + ".pt"))
            torch.save(
                model.state_dict(),
                file_name.with_suffix(file_name.suffix + ".pt"),
            )
            print("end save")
        print(
            f"Epoch {epoch:03d} "
            f"Train: Loss {train_loss:.4f}, "
            f"{Metrics().metric_format(train_pred, train_true, threshold=0.1)}\n"
            f"Validation: Loss {source_test_loss:.4f}, "
            f"{Metrics().metric_format(source_test_pred, source_test_true, threshold=0.1)}\n"
            f"Target: Loss {target_test_loss:.4f}, "
            f"{Metrics().metric_format(target_test_pred, target_test_true, threshold=0.1)}"
        )


    if config.test:

        source_test_loss, source_test_pred, source_test_true, _ = test(source_test_loader)
        best_source_test_pred, best_source_test_true = source_test_pred, source_test_true

        target_test_loss, target_test_pred, target_test_true, target_test_spec_mask = test(target_test_loader)
        best_target_test_pred, best_target_test_true = target_test_pred, target_test_true

        best_target_delta_y = (target_test_pred - target_test_true)[target_test_true > 0.1]


    draw_histogram(
        best_target_delta_y, file_name.with_suffix(file_name.suffix + ".specific.png")
    )
    print(
        (
            f"Best Target Test:"
            f"{Metrics().metric_format(best_target_test_pred, best_target_test_true, threshold=0.1)}"
        )
    )

    with file_name.with_suffix(file_name.suffix + ".result_specific.json").open(mode="w") as fp:
        result = {
            "num_source_domains": len(source_train_loaders),
            "num_test_domains": len(target_test_loaders),
            "best_source_test": Metrics().calc_metric(
                best_source_test_pred, best_source_test_true, threshold=0.1
            ),
            "target_test": Metrics().calc_metric(
                best_target_test_pred, best_target_test_true, threshold=0.1
            ),
            "target_test_spec": Metrics().calc_metric(
                best_target_test_pred[best_target_test_specmask], best_target_test_true[best_target_test_specmask], threshold=0.1
            )
        }
        json.dump(result, fp, indent=4)