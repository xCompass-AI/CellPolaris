import argparse
import json
import warnings
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
    get_data_from_loaders,
    infinite_loader_with_domain_index,
)

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
    args.add_argument(
        "--mode", type=str, default="same_tissue", help="same period or same tissue"
    )
    args.add_argument("--select_period_or_tissue", type=str, default="liver")
    args.add_argument(
        "--dg_loss_type",
        type=lambda x: None if x == "None" else x,
        default="graph_mixup",
    )
    args.add_argument("--dg_loss_weight", type=float, default=1)
    args.add_argument("--ignore_regression", action="store_true", default=False)
    args.add_argument("--mixup_alpha", type=float, default=0.2)
    args.add_argument("--model", type=str, default="ncf")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--batch_size", type=int, default=2560)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--weight_decay", type=float, default=0.01)
    args.add_argument("--epoch", type=int, default=50)
    args.add_argument("--device", type=str, default="cuda:1")
    args.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test the performance of model.",
    )
    config = args.parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if config.dg_loss_type == "None":
        config.dg_loss_type = None
    if config.dg_loss_type is None or "mixup" not in config.dg_loss_type:
        config.ignore_regression = False
        print("Ignore regression loss: False")
    dataset = GRNPredictionDataset()
    all_train_loaders, all_test_loaders = dataset.get_loaders(config.batch_size)

    print("Getting required loaders...")
    if config.mode == "same_tissue":
        select_tissue = config.select_period_or_tissue
        train_loaders = all_train_loaders["bulk_mouse"][select_tissue]
        test_loaders = all_test_loaders["bulk_mouse"][select_tissue]
    elif config.mode == "same_period":
        select_period = config.select_period_or_tissue
        train_loaders, test_loaders = {}, {}
        for tissue, data in all_train_loaders["bulk_mouse"].items():
            if select_period in data.keys():
                train_loaders[tissue] = data[select_period]
  
                test_loaders[tissue] = all_test_loaders["bulk_mouse"][tissue][select_period]

    else:
        raise NotImplementedError
    del (
        dataset.datasets,
        all_train_loaders,
        all_test_loaders,
    )
    print("Done.")

    for name, loader in train_loaders.items():
        print(name)

        source_train_loaders = [
            loader for _name, loader in train_loaders.items() if _name != name
        ]
        source_test_loaders = [
            loader for _name, loader in test_loaders.items() if _name != name
        ]
        target_train_loader = loader
        target_test_loader = test_loaders[name]
        source_min_n_iter = min(len(x) for x in source_train_loaders)
        source_train_loaders = [
            infinite_loader_with_domain_index(x, index)
            for index, x in enumerate(source_train_loaders)
        ]

        criterion = torch.nn.MSELoss()
        seed_everything(config.seed)
        link_prediction_model = build_model(config.model, dataset.num_nodes)
        model = TRModel(
            link_prediction_model,
            config.dg_loss_type,
            num_sourcedomains=len(source_train_loaders),
            mixup_alpha=config.mixup_alpha,
            ignore_regression=config.ignore_regression,
        ).to(device)
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epoch
        )

        base = Path(__file__).parent / f"result/{config.mode}_generalization"
        file_name = base / f"{config.model}/{config.dg_loss_type}/"
        if config.mode == "same_tissue":
            file_name /= f"{select_tissue}_{name}"
        else:
            file_name /= f"{name}_{select_period}"
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
            source_test_loaders, batch_size=10 * config.batch_size
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

            source_test_loss, source_test_pred, source_test_true = test(
                source_test_loader
            )

            target_test_loss, target_test_pred, target_test_true = test(
                target_test_loader
            )
            scheduler.step()

            if source_test_loss < best_source_test_loss:
                best_source_test_loss = source_test_loss
   
                best_target_delta_y = (target_test_pred - target_test_true)[target_test_true > 0.1]
                best_target_test_pred, best_target_test_true = target_test_pred, target_test_true
                best_source_test_pred, best_source_test_true = source_test_pred, source_test_true

                torch.save(
                    model.state_dict(),
                    file_name.with_suffix(file_name.suffix + ".pt"),
                )
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
        with file_name.with_suffix(file_name.suffix + ".result.json").open(
            mode="w"
        ) as fp:
            result = {
                "best_source_test": Metrics().calc_metric(
                    best_source_test_pred, best_source_test_true, threshold=0.1
                ),
                "target_test": Metrics().calc_metric(
                    best_target_test_pred, best_target_test_true, threshold=0.1
                ),
            }
            json.dump(result, fp, indent=4)
