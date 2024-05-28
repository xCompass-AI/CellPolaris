import argparse
import json
from pathlib import Path

import torch
from torch_geometric import seed_everything
from tqdm import tqdm

from load_dataset import GRNPredictionDataset
from models import TRModel, build_model
from utils import (
    Metrics,
    build_chain_loader,
    draw_histogram,
    get_cross_validation_mask,
    get_data_from_loaders,
    infinite_loader_with_domain_index,
)
import warnings

warnings.filterwarnings("error")


def train(loader_iters, n_iter=10, dg_loss_weight=1, device="cpu"):
    model.train()

    total_loss = 0
    y_pred, y_true = [], []
    for _ in range(n_iter):
        x0_list, x1_list, edges_list, y_list, tissue_index_list = get_data_from_loaders(
            loader_iters, device
        )
        optimizer.zero_grad()
        out, transfer_loss = model(
            x0_list, x1_list, edges_list, tissue_index_list, y_list
        )
        y = torch.cat(y_list, dim=0)

        if not config.ignore_regression:
            regression_loss = criterion(out, y)
        else:
            regression_loss = 0


        if transfer_loss is not None:
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
    y_pred, y_true = [], []
    for x0, x1, edges, y in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        edges = edges.t().to(device)
        y = y.to(device)

        out, _ = model.link_prediction_model(x0, x1, edges)
        loss = criterion(out, y)

        total_loss += float(loss) * len(y)
        y_pred.append(out.cpu().detach())
        y_true.append(y.cpu().detach())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    return total_loss / y_pred.shape[0], y_pred, y_true


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--species", nargs="+", type=str, default=["sc_human"])
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
    args.add_argument("--device", type=str, default="cuda:1")
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
                if spe == "sc_human":
                    if config.append_bulk:
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
                elif spe == "sc_mouse":
                    if config.append_bulk:
                        bulk_train_loaders = all_train_loaders["bulk_mouse"]
                        bulk_test_loaders = all_test_loaders["bulk_mouse"]      
                    train_loaders_tmp = all_train_loaders[spe]
                    test_loader_tmp = all_test_loaders[spe]
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

    # --------
    if config.num_source is not None:
        print(f"Tuncate source domain to {config.num_source}.")
        source_train_loaders = source_train_loaders[: config.num_source]
    # --------
    source_min_n_iter = min(len(x) for x in source_train_loaders)
    source_train_loaders = [
        infinite_loader_with_domain_index(x, index)
        for index, x in enumerate(source_train_loaders)
    ]

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
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

    base = Path(__file__).parent / "result/multi_to_multi_generalization"
    if config.dg_loss_type == 'graph_mixup':
        file_name = (
            base / f"{config.model}/{config.dg_loss_type}/"
            f"{config.species}/mixupalpha{config.mixup_alpha}_topratio{config.top_ratio}/fold_{config.fold}"
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

    print("Building chain loaders for accelerating ...")
    source_test_loader = build_chain_loader(
        source_test_loaders, batch_size=config.batch_size
    )
    target_test_loader = build_chain_loader(
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

        source_test_loss, source_test_pred, source_test_true = test(source_test_loader)

        target_test_loss, target_test_pred, target_test_true = test(target_test_loader)
        scheduler.step()

        if source_test_loss < best_source_test_loss:
            best_source_test_loss = source_test_loss

            best_target_delta_y = (target_test_pred - target_test_true)[target_test_true > 0.1] #0.1
            best_target_test_pred, best_target_test_true = target_test_pred, target_test_true
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

        source_test_loss, source_test_pred, source_test_true = test(source_test_loader)
        best_source_test_pred, best_source_test_true = source_test_pred, source_test_true

        target_test_loss, target_test_pred, target_test_true = test(target_test_loader)
        best_target_test_pred, best_target_test_true = target_test_pred, target_test_true

        best_target_delta_y = (target_test_pred - target_test_true)[target_test_true > 0.1]


    draw_histogram(
        best_target_delta_y, file_name.with_suffix(file_name.suffix + ".png")
    )
    print(
        (
            f"Best Target Test:"
            f"{Metrics().metric_format(best_target_test_pred, best_target_test_true, threshold=0.1)}"
        )
    )
    with file_name.with_suffix(file_name.suffix + ".result.json").open(mode="w") as fp:
        result = {
            "num_source_domains": len(source_train_loaders),
            "num_test_domains": len(target_test_loaders),
            "best_source_test": Metrics().calc_metric(
                best_source_test_pred, best_source_test_true, threshold=0.1
            ),
            "target_test": Metrics().calc_metric(
                best_target_test_pred, best_target_test_true, threshold=0.1
            ),
        }
        json.dump(result, fp, indent=4)
