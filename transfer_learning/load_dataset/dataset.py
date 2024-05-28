import copy
from collections import defaultdict
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from torch_geometric.utils import negative_sampling, structured_negative_sampling
from .load_each_peca import AllPECALoader


class GRNPredictionDataset:
    def __init__(self, root="dataset"):
        self.root = Path(root)
        self.processed_name = "processed_dataset.pt"  # processed_dataset_human_fly.pt | processed_dataset_human_zebrafish.pt
        print(f"Dataset version: {self.processed_name}.")
        if (self.root / self.processed_name).exists():
            print("Loading preprocessed data...")
            self.datasets = torch.load(self.root / self.processed_name)
            print("Done.")
        else:
            print("Preprocessing raw data...")
            self.peca_graph = AllPECALoader()
            self.num_nodes = self.peca_graph.num_nodes
            self.datasets = self.get_dataset()
            self.datasets["num_nodes"] = self.num_nodes

            ## human and mouse
            self.datasets["human_to_mouse_mapping"] = self.peca_graph.human_to_mouse_mapping
            # self.datasets["mouse_to_human_mapping"] = {
            #     v: k for k, v in self.peca_graph.human_to_mouse_mapping.items()
            # }

            ## human with fly
            # self.datasets["human_to_fly_mapping"] = self.peca_graph.human_to_fly_mapping
            ## human with zebrafish
            # self.datasets["human_to_zebrafish_mapping"] = self.peca_graph.human_to_zebrafish_mapping
            
            self.datasets["gene_names"] = self.peca_graph.gene_names
            self.datasets["preserve_gene_mask"] = self.peca_graph.preserve_gene_mask
            self.datasets["all_tf_names"] = self.peca_graph.all_tf_names
            self.datasets["all_tg_names"] = self.peca_graph.all_tg_names
            self.datasets["specific_tf_names"] = self.peca_graph.specific_tf_names
            self.datasets["specific_tg_names"] = self.peca_graph.specific_tg_names
            print("Saving preprocessed data...")
            torch.save(self.datasets, self.root / self.processed_name)
            print("Done.")
        self.num_nodes = self.datasets["num_nodes"]
        self.preserve_gene_mask = self.datasets["preserve_gene_mask"]
        ## human and mouse
        self.human_to_mouse_mapping = self.datasets["human_to_mouse_mapping"]
        # if "mouse_to_human_mapping" not in self.datasets:
        #     self.mouse_to_human_mapping = {
        #         v: k for k, v in self.human_to_mouse_mapping.items()
        #     }
        # else:
        #     self.mouse_to_human_mapping = self.datasets["mouse_to_human_mapping"]

        ## human with fly
        # self.human_to_fly_mapping = self.datasets["human_to_fly_mapping"]

        ## human with zebrafish
        # self.human_to_zebrafish_mapping = self.datasets["human_to_zebrafish_mapping"]
        
        self.gene_names = self.datasets["gene_names"]
        self.all_tf_names = self.datasets["all_tf_names"]
        self.all_tg_names = self.datasets["all_tg_names"]
        self.specific_tf_names = self.datasets["specific_tf_names"]
        self.specific_tg_names = self.datasets["specific_tg_names"]
        self.all_tf_mask, self.all_tg_mask = self.get_tf_tg_mask(
            self.all_tf_names, self.all_tg_names
        )
        self.specific_tf_mask, self.specific_tg_mask = {}, {}
        for species in self.specific_tf_names.keys():
            (
                self.specific_tf_mask[species],
                self.specific_tg_mask[species],
            ) = self.get_tf_tg_mask(
                self.specific_tf_names[species], self.specific_tg_names[species]
            )
        self.specific_tf_mask["sc_mouse"] = (
            self.specific_tf_mask["sc_mouse"]
            | self.specific_tf_mask["sc_mouse_hsc"]
            | self.specific_tf_mask["sc_mouse_placenta"]
        )
        self.specific_tf_mask["bulk_mouse"] = (
            self.specific_tf_mask["bulk_mouse"]
            | self.specific_tf_mask["bulk_mouse_master_tf"]
        )
        self.specific_tg_mask["sc_mouse"] = (
            self.specific_tg_mask["sc_mouse"]
            | self.specific_tg_mask["sc_mouse_hsc"]
            | self.specific_tg_mask["sc_mouse_placenta"]
        )
        self.specific_tg_mask["bulk_mouse"] = (
            self.specific_tg_mask["bulk_mouse"]
            | self.specific_tg_mask["bulk_mouse_master_tf"]
        )

        del self.datasets["num_nodes"]
        del self.datasets["gene_names"]
        del self.datasets["preserve_gene_mask"]
        del self.datasets["all_tf_names"]
        del self.datasets["all_tg_names"]
        del self.datasets["specific_tf_names"]
        del self.datasets["specific_tg_names"]
        ## human and mouse
        del self.datasets["human_to_mouse_mapping"]
        # del self.datasets["mouse_to_human_mapping"]

        ## human with fly
        # del self.datasets["human_to_fly_mapping"]
        
        ## human with zebrafish
        # del self.datasets["human_to_zebrafish_mapping"]


    def get_dataset(self):
        print("Building datasets...")
        datasets = {}
        datasets["bulk_mouse"] = defaultdict(dict)
        for name, tissue in self.peca_graph.edge_collections["bulk_mouse"].items():
            for period, edge_index in tissue.items():
                edge_weight, rna_seq, rna_seq_mask = (
                    self.peca_graph.edge_weight_collections["bulk_mouse"][name][period],
                    self.peca_graph.rna_seq_collections["bulk_mouse"][name][period][0],
                    self.peca_graph.rna_seq_collections["bulk_mouse"][name][period][1],
                )
                datasets["bulk_mouse"][name][period] = self.get_full_dataset(
                    edge_index,
                    edge_weight,
                    rna_seq,
                    rna_seq_mask,
                    self.num_nodes,
                    neg_sampling_ratio=3,
                )
        for species in self.peca_graph.species[1:]:
            datasets[species] = {}
            for name, edge_index in self.peca_graph.edge_collections[species].items():
                edge_weight, rna_seq, rna_seq_mask = (
                    self.peca_graph.edge_weight_collections[species][name],
                    self.peca_graph.rna_seq_collections[species][name][0],
                    self.peca_graph.rna_seq_collections[species][name][1],
                )
                datasets[species][name] = self.get_full_dataset(
                    edge_index,
                    edge_weight,
                    rna_seq,
                    rna_seq_mask,
                    self.num_nodes,
                    neg_sampling_ratio=6,
                )

        print("Done.")
        return datasets

    def get_loaders(
        self,
        batch_size=32,
        preserve_only_pos_edges=False,
        preserve_only_neg_edges=False,
    ):
        r"""
        Get data loaders for training and testing.

        Args:
            batch_size (int): batch size.
            preserve_only_pos_edges (bool): For domain adaptation, we only preserve edges
                that have existed in the previous GRN.
            preserve_only_neg_edges (bool): For domain adaptation, we only preserve
                negative edges.
        """
        if preserve_only_pos_edges and preserve_only_neg_edges:
            raise RuntimeWarning(
                "preserve_only_true_edges and preserve_only_neg_edges cannot be both True."
                "We will only preserve true edges."
            )
        if not preserve_only_pos_edges and not preserve_only_neg_edges:
            print("Building loaders...")
        else:
            print("Building loaders for domain adaptation...")
        train_loaders, test_loaders = {}, {}
        train_loaders["bulk_mouse"] = defaultdict(dict)
        test_loaders["bulk_mouse"] = defaultdict(dict)
        for name, tissue in self.datasets["bulk_mouse"].items():
            for period, dataset in tissue.items():
                if preserve_only_pos_edges:
                    train_dataset = keep_only_positive_edges(dataset[0])
                    test_dataset = keep_only_positive_edges(dataset[1])
                elif preserve_only_neg_edges:
                    train_dataset = keep_only_negative_edges(dataset[0])
                    test_dataset = keep_only_negative_edges(dataset[1])
                else:
                    train_dataset, test_dataset = dataset[0], dataset[1]
                train_loaders["bulk_mouse"][name][period] = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                )
                test_loaders["bulk_mouse"][name][period] = DataLoader(
                    test_dataset,
                    batch_size=3000,
                    shuffle=False,
                    num_workers=1,
                )
        for species in set(self.datasets.keys()) - set(["bulk_mouse"]):
            train_loaders[species], test_loaders[species] = {}, {}
            for name, dataset in self.datasets[species].items():
                if preserve_only_pos_edges:
                    train_dataset = keep_only_positive_edges(dataset[0])
                    test_dataset = keep_only_positive_edges(dataset[1])
                elif preserve_only_neg_edges:
                    train_dataset = keep_only_negative_edges(dataset[0])
                    test_dataset = keep_only_negative_edges(dataset[1])
                else:
                    train_dataset, test_dataset = dataset[0], dataset[1]
                train_loaders[species][name] = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                )
                test_loaders[species][name] = DataLoader(
                    test_dataset,
                    batch_size=3000,
                    shuffle=False,
                    num_workers=1,
                )

        print("Done.")
        return train_loaders, test_loaders

    def get_full_dataset(
        self,
        edge_index,
        edge_weight,
        rna_seq,
        rna_seq_mask,
        num_nodes,
        neg_sampling_ratio=1,
        neg_sampling_mode="tftg",
        min_pos_edge_weight=0.15,
    ):
        r"""
        Get train and test dataset.

        Procedure:
            1. negtive sampling
            2. drop unimportant edges according to min_pos_edge_weight
            3. concat pos and neg edges
        """
        assert neg_sampling_mode in ["global", "tftg"]
        seed_everything(0)
        pos_edge_index = edge_index
        pos_edge_weight = edge_weight
        if neg_sampling_mode == "global":
            neg_edge_index = negative_sampling(
                pos_edge_index,
                num_nodes,
                num_neg_samples=pos_edge_index.shape[1] * neg_sampling_ratio,
            )
        else:
            neg_sampling_result = []
            for _ in range(int(neg_sampling_ratio)):
                result = structured_negative_sampling(
                    pos_edge_index,
                    num_nodes=num_nodes,
                    contains_neg_self_loops=False,
                )
                neg_sampling_result.append(torch.stack((result[0], result[2])))
            neg_edge_index = torch.cat(neg_sampling_result, dim=1)

        neg_edge_weight = torch.FloatTensor([-0.05] * neg_edge_index.shape[1])

        mask = pos_edge_weight > min_pos_edge_weight
        pos_edge_index = pos_edge_index[:, mask]
        pos_edge_weight = pos_edge_weight[mask]

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_weight = torch.cat([pos_edge_weight, neg_edge_weight], dim=0)

        mask = torch.arange(edge_weight.shape[0])
        train_mask, test_mask = train_test_split(mask, test_size=0.2, random_state=42)
        train_dataset = PECADataset(
            edge_index[:, train_mask], edge_weight[train_mask], rna_seq, rna_seq_mask
        )
        test_dataset = PECADataset(
            edge_index[:, test_mask], edge_weight[test_mask], rna_seq, rna_seq_mask
        )
        return train_dataset, test_dataset

    def get_tf_tg_mask(self, tf_names, tg_names):
        tf_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        tg_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        for i, gene_name in enumerate(self.gene_names):
            if gene_name in tf_names:
                tf_mask[i] = True
            if gene_name in tg_names:
                tg_mask[i] = True
        return tf_mask, tg_mask


class PECADataset:
    def __init__(self, edge_index, edge_weight, rna_seq, rna_seq_mask):
        self.edge_index = edge_index.transpose(0, 1)
        self.edge_weight = edge_weight
        self.x0 = rna_seq[edge_index[0]]
        self.x1 = rna_seq[edge_index[1]]
        self.len = edge_index.shape[1]
        self.rna_seq = rna_seq
        self.rna_seq_mask = rna_seq_mask

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.x0[index],
            self.x1[index],
            self.edge_index[index],
            self.edge_weight[index],
        )


def keep_only_positive_edges(dataset):
    r"""
    Keep only positive edges in the dataset.

    Args:
        dataset (PECADataset): dataset.
    """
    dataset = copy.deepcopy(dataset)
    mask = dataset.edge_weight > 0.1
    dataset.edge_index = dataset.edge_index[mask]
    dataset.edge_weight = dataset.edge_weight[mask]
    dataset.x0 = dataset.x0[mask]
    dataset.x1 = dataset.x1[mask]
    dataset.len = dataset.x0.shape[0]
    return dataset


def keep_only_negative_edges(dataset):
    r"""
    Keep only negative edges in the dataset.

    Args:
        dataset (PECADataset): dataset.
    """
    dataset = copy.deepcopy(dataset)
    mask = dataset.edge_weight < 0
    dataset.edge_index = dataset.edge_index[mask]
    dataset.edge_weight = dataset.edge_weight[mask]
    dataset.x0 = dataset.x0[mask]
    dataset.x1 = dataset.x1[mask]
    dataset.len = dataset.x0.shape[0]
    return dataset
