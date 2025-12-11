import argparse
from itertools import chain
from pathlib import Path
import torch
from generate_grn import dump_generated_grn, evaluate_generated_grn, generate_grn
from load_dataset import GRNPredictionDataset
from models import TRModel, build_model
from utils import get_cross_validation_mask

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--species", nargs="+", type=str, default=["bulk_mouse","bulk_human"])
    args.add_argument("--tissue_list", nargs="+", type=str, default=["heart", "kidney", "spleen", "liver", "lung"], help="source tissues from human and mouse")
    args.add_argument("--target",type=str, default="kidney", help="this is from human bulk, for cross species")
    args.add_argument("--fold", "--k-of-4-fold", type=int, default=2)
    args.add_argument("--dg_loss_type", type=str, default=None)#"graph_extramixup"
    args.add_argument("--num_source", type=int, default=None)
    args.add_argument("--model", type=str, default="ncf")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--device", type=str, default="cuda:3")
    args.add_argument("--dg_loss_weight", type=float, default=1)
    args.add_argument("--ignore_regression", action="store_true", default=False)
    args.add_argument("--mixup_alpha", type=float, default=0.2)
    args.add_argument("--top_ratio", type=float, default=0.11, help="if dg_loss_type is graph-based, this is a hype-parameter")
    args.add_argument("--batch_size", type=int, default=1280)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--weight_decay", type=float, default=0.01)
    args.add_argument("--epoch", type=int, default=100)


    config = args.parse_args()
    if config.dg_loss_type == "None":
        config.dg_loss_type = None
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dataset = GRNPredictionDataset()

    all_train_datasets = {}
    all_test_datasets = {}

    if len(config.species)>1:
        if "bulk_mouse" in dataset.datasets.keys():
            all_train_datasets["bulk_mouse"], all_test_datasets["bulk_mouse"] = [], []
            select_tissue = ["cerebellum", "spleen", "liver", "heart", "kidney", "lung"]
            for tissue_name, tissue in dataset.datasets["bulk_mouse"].items():
                if tissue_name in select_tissue:
                    period_name = list(tissue.keys())[0]
                    data = tissue[period_name]
                    # period_name, loader = next(iter(tissue.items()))
                    all_train_datasets["bulk_mouse"].append((f"{tissue_name}", data[0]))
                    all_test_datasets["bulk_mouse"].append((f"{tissue_name}", data[1]))
        if "bulk_human" in dataset.datasets.keys():
            all_train_datasets["bulk_human"], all_test_datasets["bulk_human"] = [], []
            select_tissue = ["cerebellum", "spleen", "right_lobe_of_liver", "Fetal_Heart", "Fetal_Kidney", "Fetal_Lung"]
            for name, data in dataset.datasets["bulk_human"].items():
                if name in select_tissue:
                    all_train_datasets["bulk_human"].append((f"{name}", data[0]))
                    all_test_datasets["bulk_human"].append((f"{name}", data[1]))
        all_train_datasets["bulk_mouse"] = { t[0]: t[1] for t in all_train_datasets["bulk_mouse"] }
        all_test_datasets["bulk_mouse"] = { t[0]: t[1] for t in all_test_datasets["bulk_mouse"] }
        all_train_datasets["bulk_human"] = { t[0]: t[1] for t in all_train_datasets["bulk_human"] }
        all_test_datasets["bulk_human"] = { t[0]: t[1] for t in all_test_datasets["bulk_human"] }
        print("Have got the train and test set.")
    else:
        if "bulk_mouse" in dataset.datasets.keys():
            all_train_datasets["bulk_mouse"], all_test_datasets["bulk_mouse"] = [], []
            for tissue_name, tissue in dataset.datasets["bulk_mouse"].items():
                for period_name, data in tissue.items():
                    all_train_datasets["bulk_mouse"].append(
                        (f"{tissue_name}", data[0])
                    )
                    all_test_datasets["bulk_mouse"].append(
                        (f"{tissue_name}", data[1])
                    )
        for species in ["bulk_human", "sc_human", "sc_mouse"]:
            if species in dataset.datasets.keys():
                all_train_datasets[species], all_test_datasets[species] = [], []
                for name, data in dataset.datasets[species].items():
                    all_train_datasets[species].append((f"{name}", data[0]))
                    all_test_datasets[species].append((f"{name}", data[1]))

        for name, data in dataset.datasets["bulk_mouse_master_tf"].items():
            all_train_datasets["bulk_mouse"].append((f"{name}", data[0]))
            all_test_datasets["bulk_mouse"].append((f"{name}", data[1]))

    target_train_datasets, target_test_datasets = [], []
    target_species = []
    source_train_datasets, source_test_datasets = [], []
    source_species = []

    species = config.species
    if all(x in ["bulk_mouse", "bulk_human", "sc_mouse", "sc_human"] for x in species):
        if len(species)>1:
            if species == ["sc_mouse", "sc_human"]:
                for spe in species:
                    if spe == "sc_human":
                        if config.append_bulk:
                            bulk_train_loaders = all_train_datasets["bulk_human"]
                            bulk_test_loaders = all_test_datasets["bulk_human"]

                        train_loaders_tmp = all_train_datasets[spe]
                        test_loaders_tmp = all_test_datasets[spe]
                        mask = get_cross_validation_mask(len(train_loaders_tmp), config.seed, fold=4)
                        train_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()
                        test_mask = mask[config.fold].tolist()
                        for index in train_mask:
                            source_train_datasets.append(train_loaders_tmp[index])
                            source_test_datasets.append(test_loaders_tmp[index])
                        for index in test_mask:
                            target_train_datasets.append(train_loaders_tmp[index])
                            target_test_datasets.append(test_loaders_tmp[index])
                        if config.append_bulk:
                            source_train_datasets += bulk_train_loaders
                            source_test_datasets += bulk_test_loaders
                        del (
                            dataset.datasets,
                            all_train_datasets,
                            all_test_datasets,
                            train_loaders_tmp,
                            test_loaders_tmp,
                            )
                    elif spe == "sc_mouse":
                        if config.append_bulk:
                            bulk_train_loaders = all_train_datasets["bulk_mouse"]
                            bulk_test_loaders = all_test_datasets["bulk_mouse"]      
                        train_loaders_tmp = all_train_datasets[spe]
                        test_loader_tmp = all_test_datasets[spe]
                        mask = get_cross_validation_mask(len(train_loaders_tmp), config.seed, fold=4)
                        train_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()
                        for index in train_mask:
                            source_train_datasets.append(train_loaders_tmp[index])
                        if config.append_bulk:
                            source_train_datasets += bulk_train_loaders
                            source_test_datasets += bulk_test_loaders

                        del (
                            dataset.datasets,
                            all_train_datasets,
                            all_test_datasets,
                            train_loaders_tmp,
                            test_loaders_tmp,
                        )
            elif species == ["bulk_mouse", "bulk_human"]:
                strong_related_source = None
                tissue_index_map = []
                target = config.target.lower()
                for key, loader in all_train_datasets["bulk_human"].items():
                    if target in key.lower():
                        target_train_datasets.append((f"{key}", all_train_datasets["bulk_human"][key]))
                        target_test_datasets.append((f"{key}", all_test_datasets["bulk_human"][key]))
                        target_species.append("bulk_human")
                        break
                
                sources = [t for t in config.tissue_list if t != config.target]

                tissue_index_map = {
                                    t.lower(): {"human": None, "mouse": None}
                                    for t in config.tissue_list if t != config.target
                                    }
                
                for spe in species:
                    if spe == "bulk_human":
                        for key, loader in all_train_datasets["bulk_human"].items():
                            key_lower = key.lower()
                            for tissue in sources:
                                tissue_lower = tissue.lower()
                                if tissue_lower in key_lower:
                                    
                                    source_train_datasets.append((f"{key}", all_train_datasets["bulk_human"][key]))
                                    source_test_datasets.append((f"{key}", all_test_datasets["bulk_human"][key]))
                                    source_species.append(spe)
                                    if tissue_index_map[tissue_lower]["human"] is None:
                                        tissue_index_map[tissue_lower]["human"] = len(source_train_datasets) - 1
                                    break 

                    elif spe == "bulk_mouse":
                        for key, loader in all_train_datasets["bulk_mouse"].items():
                            key_lower = key.lower()
                            for tissue in config.tissue_list:
                                tissue_lower = tissue.lower()
                                if tissue_lower in key_lower:
                                    source_train_datasets.append((f"{key}", all_test_datasets["bulk_mouse"][key]))
                                    source_test_datasets.append((f"{key}", all_test_datasets["bulk_mouse"][key]))
                                    source_species.append(spe)
                               
                                    if tissue_lower == target:
                                        strong_related_source = len(source_train_datasets) - 1
                          
                                    elif tissue_lower in tissue_index_map and tissue_index_map[tissue_lower]["mouse"] is None:
                                        tissue_index_map[tissue_lower]["mouse"] = len(source_train_datasets) - 1
                                    break
                
                del (
                    dataset.datasets,
                    all_train_datasets,
                    all_test_datasets,
                )       
            print("Have got source and target set.")
        else:
            all_train_datasets = all_train_datasets[config.species]
            all_test_datasets = all_test_datasets[config.species]
            mask = get_cross_validation_mask(len(all_train_datasets), config.seed, fold=4)
            source_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()
            target_mask = mask[config.fold].tolist()

            for index in source_mask:
                source_train_datasets.append(all_train_datasets[index])
                source_test_datasets.append(all_test_datasets[index])
                source_species.append(config.species)
            for index in target_mask:
                target_train_datasets.append(all_train_datasets[index])
                target_test_datasets.append(all_test_datasets[index])
                target_species.append(config.species)

            if config.species == "sc_mouse" and config.fold == 0:
                for name, data in chain(
                    dataset.datasets["sc_mouse_hsc"].items(),
                    dataset.datasets["sc_mouse_placenta"].items(),
                ):
                    target_train_datasets.append((f"{name}", data[0]))
                    target_test_datasets.append((f"{name}", data[1]))
                    target_species.append("sc_mouse")

    if config.num_source is not None:
        print(f"Tuncate source domain to {config.num_source}.")
        source_train_datasets = source_train_datasets[: config.num_source]

    link_prediction_model = build_model(config.model, dataset.num_nodes)

    model = TRModel(
        link_prediction_model,
        config.dg_loss_type,
        num_sourcedomains=len(source_train_datasets),
        mixup_alpha=config.mixup_alpha,
        ignore_regression=config.ignore_regression,
        top_ratio = config.top_ratio,
        strong_related_source = strong_related_source,
        tissue_index_map = tissue_index_map,
    ).to(device)

    base = Path(__file__).parent / f"result_{config.tissue_list}/multi_to_multi_generalization"
    file_name = (
        base / f"{config.model}/{config.dg_loss_type}/"
        f"{config.species}/target_{config.target}"
    )
    if config.num_source is not None:
        file_name = (
            file_name.parent / f"saturation/fold_{config.fold}_{config.num_source}"
        )

    model_path = file_name.with_suffix(file_name.suffix + ".pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.link_prediction_model
    model.eval()

    base = (
        Path(__file__).parent
        / f"result_{config.tissue_list}/multi_to_multi_generalization/"
        / f"{config.model}/{config.dg_loss_type}/{config.species}/generated_grn"
    )
    base.mkdir(parents=True, exist_ok=True)
    for (file_name, train_dataset), (_, test_dataset), species in zip(
        target_train_datasets[:], target_test_datasets[:], target_species[:]
    ):

        print(file_name)
        rna_seq = train_dataset.rna_seq
        rna_seq_mask = train_dataset.rna_seq_mask
        generated_edge_index, generated_edge_weight = generate_grn(
            model,
            rna_seq.to(device),
            rna_seq_mask.to(device),
            dataset.specific_tf_mask[species],
            dataset.specific_tg_mask[species],
        )

        f = base / f"{species}_{file_name}_generated_grn"
        grn_save_path = f.with_suffix(f.suffix + ".txt")

        true_edge_index = torch.cat(
            [
                train_dataset.edge_index.t(),
                test_dataset.edge_index.t(),
            ],
            dim=1,
        )
        true_edge_weight = torch.cat(
            [
                train_dataset.edge_weight,
                test_dataset.edge_weight,
            ],
            dim=0,
        )
        mask = true_edge_weight > 0
        true_edge_index = true_edge_index[:, mask]
        true_edge_weight = true_edge_weight[mask]

        results = evaluate_generated_grn(
            generated_edge_index,
            generated_edge_weight,
            true_edge_index,
            true_edge_weight,
            f,
        )
        dump_generated_grn(
            generated_edge_index,
            generated_edge_weight,
            save_to_path=grn_save_path,
            gene_names=dataset.gene_names,
            out_is_human=True if "human" in config.species else False,
            human_to_mouse_mapping=dataset.human_to_mouse_mapping,
            cross_species=False, 
        )
