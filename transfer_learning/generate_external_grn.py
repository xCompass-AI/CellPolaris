import argparse
from pathlib import Path

import pandas as pd
import torch

from generate_grn import dump_generated_grn, generate_grn, process_externel_rna_seq
from load_dataset import GRNPredictionDataset
from models import TRModel, build_model
from utils import get_cross_validation_mask

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--species", type=str, default="bulk_human", help="Source species.")
    args.add_argument("--fold", "--k-of-4-fold", type=int, default=0)
    args.add_argument("--transfer_loss_type", type=str, default="graph_mixup")
    args.add_argument("--model", type=str, default="ncf")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--device", type=str, default="cuda:1")
    args.add_argument("--top_ratio", type=float, default=0.2, help="if dg_loss_type is graph-based, this is a hype-parameter")
    args.add_argument(
        "--rna_seq_path",
        nargs="+",
        help="Path to RNA-seq data.",
        default=[
            "/home/ict/tiny_model_transfer/GRNPredict_tr/dataset/carT/new_RNA/T52.txt"
        ],
    )

    config = args.parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dataset = GRNPredictionDataset()

    mask = get_cross_validation_mask(
        len(dataset.datasets[config.species]), config.seed, fold=4
    )
    source_mask = torch.cat(mask[: config.fold] + mask[config.fold + 1 :]).tolist()

    link_prediction_model = build_model(config.model, dataset.num_nodes)
    model = TRModel(
        link_prediction_model,
        config.transfer_loss_type,
        num_sourcedomains=len(source_mask),
        mixup_alpha=0.2,
        top_ratio = config.top_ratio,
    ).to(device)

    base = Path(__file__).parent / "result/multi_to_multi_generalization"
    file_name = (
        base / f"{config.model}/{config.transfer_loss_type}/"
        f"{config.species}/mixupalpha0.2_topratio{config.top_ratio}/fold_{config.fold}"
    )

    model_path = file_name.with_suffix(file_name.suffix + ".pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.link_prediction_model
    model.eval()

    for rna_seq_path in config.rna_seq_path:
        rna_seq_path = Path(rna_seq_path)
        if not rna_seq_path.exists():
            raise ValueError(f"{rna_seq_path} does not exist.")

        base = rna_seq_path.parent
        base /= (
            "generated_grn"
        )
        base.mkdir(parents=True, exist_ok=True)
        external_rna_seq = pd.read_csv(
            rna_seq_path, index_col=0, header=0, delimiter="\t"
        )

        is_mouse = "mouse" in config.species
        is_human = "human" in config.species

        external_rna_seq, external_rna_seq_mask = process_externel_rna_seq(
            external_rna_seq,
            dataset.gene_names,
            dataset.preserve_gene_mask,
            is_bulk="bulk_" in config.species,
            is_sc="sc_" in config.species,
            is_mouse=is_mouse,
            is_human=is_human,
            mapping=dataset.human_to_mouse_mapping,
        )
        generated_edge_index, generated_edge_weight = generate_grn(
            model,
            external_rna_seq.to(device),
            external_rna_seq_mask.to(device),
            dataset.specific_tf_mask[config.species],
            dataset.specific_tg_mask[config.species],
        )

        grn_save_path = base / f"{rna_seq_path.stem}_generated_grn.txt"
        dump_generated_grn(
            generated_edge_index,
            generated_edge_weight,
            save_to_path=grn_save_path,
            gene_names=dataset.gene_names,
            out_is_human=True
            if ("human" in config.species)
            or ("mouse" in config.species)
            else False,
            human_to_mouse_mapping=dataset.human_to_mouse_mapping,
        )
