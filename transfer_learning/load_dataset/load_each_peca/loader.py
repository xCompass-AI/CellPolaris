# -*- coding: UTF-8 -*-

from collections import defaultdict
import numpy as np
import torch
from torch_geometric.utils import (
    add_self_loops,
    remove_isolated_nodes,
    remove_self_loops,
)

from .implementation import (
    BulkHumanPECA,
    BulkMouseMasterTFPECA,
    BulkMousePECA,
    SingleCellHumanPECA,
    SingleCellMouseHSCPECA,
    SingleCellMousePECA,
    SingleCellMousePlacentaPECA,
)
from .utils import RNASeqStatisticsFeature, min_max_scale
import pandas as pd

class AllPECALoader:
    def __init__(self):
        ## human and mouse
        self.human_to_mouse_mapping = None
        ## human with fly
        # self.human_to_fly_mapping = None
        ## human with zebrafish
        # self.human_to_zebrafish_mapping = None

        print(f"Loading raw peca...")
        self.peca_processors = self.load_raw_peca()
        self.gene_names, self.gene_name_to_index = self.get_full_genes()
        (
            self.all_tf_names,
            self.all_tg_names,
            self.specific_tf_names,
            self.specific_tg_names,
        ) = self.get_tf_tg_names()

        print(f"Building edge index...")
        self.edge_collections, self.edge_weight_collections = self.build_edge_index()
        print(f"Filtering isolated nodes...")
        self.preserve_gene_mask = self.filter_unused_genes()
        self.num_preserve = self.preserve_gene_mask.sum().item()
        print(
            f"After remove isolated nodes, "
            f"there are total {self.num_preserve} genes"
        )
        print("Processing RNA-Seq data...")
        self.rna_seq_collections = self.process_rna_seq()


        df = pd.DataFrame(list(self.gene_name_to_index.items()), columns=['GeneName', 'Index'])

        ## human with fly
        # df.to_csv('human_2fly_gene_name_to_index.txt', sep='\t', index=False)
        ## human with zebrafish
        # df.to_csv('human_2zebrafish_gene_name_to_index.txt', sep='\t', index=False)


        print("Done.")

    def load_raw_peca(self):
        raw_peca = {
            "bulk_mouse": BulkMousePECA(),
            "bulk_human": BulkHumanPECA(),
            "sc_mouse": SingleCellMousePECA(),
            "sc_human": SingleCellHumanPECA(),
            "bulk_mouse_master_tf": BulkMouseMasterTFPECA(),
            "sc_mouse_hsc": SingleCellMouseHSCPECA(),
            "sc_mouse_placenta": SingleCellMousePlacentaPECA(),
        }

        ## human and mouse
        self.human_to_mouse_mapping = raw_peca["bulk_human"].human_to_mouse_mapping
        ## human with fly
        # self.human_to_fly_mapping = raw_peca["sc_human"].human_to_fly_mapping
        ## human with zebrafish        
        # self.human_to_zebrafish_mapping = raw_peca["sc_human"].human_to_zebrafish_mapping

        self.species = [
            "bulk_mouse",
            "bulk_human",
            "sc_mouse",
            "sc_human",
            "bulk_mouse_master_tf",
            "sc_mouse_hsc",
            "sc_mouse_placenta",
        ]
        return raw_peca

    def get_full_genes(self):
        raw_gene_names = []
        for species, species_data in self.peca_processors.items():
            for data in species_data:
                raw_gene_names.append(data.rna_seq.index)
        union_gene_names = set(raw_gene_names[0])
        for gene_names in raw_gene_names[1:]:
            union_gene_names = union_gene_names.union(set(gene_names))
        gene_names = sorted(list(union_gene_names))
        gene_name_to_index = {name: index for index, name in enumerate(gene_names)}
        return gene_names, gene_name_to_index

    def get_tf_tg_names(self):
        tf_names, tg_names = defaultdict(list), defaultdict(list)
        for species, species_data in self.peca_processors.items():
            for data in species_data:
                tf_names[species].append(data.tf_names)
                tg_names[species].append(data.tg_names)
        for species, species_data in tf_names.items():
            union_tf_names = set(species_data[0])
            for tf_name in species_data[1:]:
                union_tf_names = union_tf_names.union(set(tf_name))
            tf_names[species] = union_tf_names
        for species, species_data in tg_names.items():
            union_tg_names = set(species_data[0])
            for tg_name in species_data[1:]:
                union_tg_names = union_tg_names.union(set(tg_name))
            tg_names[species] = union_tg_names
        all_tf_names = None
        for species, species_data in tf_names.items():
            if all_tf_names is None:
                all_tf_names = species_data
            else:
                all_tf_names = all_tf_names.union(species_data)
        all_tg_names = None
        for species, species_data in tg_names.items():
            if all_tg_names is None:
                all_tg_names = species_data
            else:
                all_tg_names = all_tg_names.union(species_data)
        return all_tf_names, all_tg_names, tf_names, tg_names

    @property
    def num_nodes(self):
        return len(self.gene_names)

    def build_edge_index(self):
        edge_collections, edge_weight_collections = {}, {}

        edge_collections["bulk_mouse"], edge_weight_collections["bulk_mouse"] = (
            defaultdict(dict),
            defaultdict(dict),
        )
        for name, tissue in self.peca_processors["bulk_mouse"].raw_peca_data.items():
            for period, peca in tissue.items():
                edge_index, edge_weight = self._build_edge_index(peca)
                edge_collections["bulk_mouse"][name][period] = edge_index
                edge_weight_collections["bulk_mouse"][name][period] = edge_weight

        for species in self.species[1:]:
            edge_collections[species], edge_weight_collections[species] = {}, {}
            for name, peca in self.peca_processors[species].raw_peca_data.items():
                edge_index, edge_weight = self._build_edge_index(peca)
                edge_collections[species][name] = edge_index
                edge_weight_collections[species][name] = edge_weight

        return edge_collections, edge_weight_collections

    def _build_edge_index(self, peca):
        gene_names = set(self.gene_names)
        grn_data = peca.grn_data[peca.grn_data.iloc[:, 0].isin(gene_names)]
        grn_data = grn_data[grn_data.iloc[:, 1].isin(gene_names)]
        edge_index = grn_data.iloc[:, :2].values
        edge_index = [
            [self.gene_name_to_index[gene_name] for gene_name in edge]
            for edge in edge_index
        ]
        edge_weight = grn_data.iloc[:, 2].values
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_weight = min_max_scale(edge_weight.log()) * 1.05
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def filter_unused_genes(self):
        all_mask = torch.zeros(self.num_nodes)


        for tissue in self.edge_collections["bulk_mouse"].values():
            for edge_index in tissue.values():
                _, _, mask = remove_isolated_nodes(edge_index, num_nodes=self.num_nodes)
                all_mask = torch.logical_or(all_mask, mask)


        for species in self.species[1:]:
            for edge_index in self.edge_collections[species].values():
                _, _, mask = remove_isolated_nodes(edge_index, num_nodes=self.num_nodes)
                all_mask = torch.logical_or(all_mask, mask)

        return all_mask

    def process_rna_seq(self):
        rna_seq_feature = RNASeqStatisticsFeature()
        xs = {}


        xs["bulk_mouse"] = defaultdict(dict)
        for name, tissue in self.peca_processors["bulk_mouse"].raw_peca_data.items():
            for period, peca in tissue.items():
                print(name, period, end="\t")
                rna_seq = np.zeros(len(self.gene_names)) - 1
                for index, (gene_name, is_preserve) in enumerate(
                    zip(self.gene_names, self.preserve_gene_mask)
                ):
                    if is_preserve and gene_name in peca.rna_seq.index:
                        rna_seq[index] = peca.rna_seq.loc[gene_name]
                rna_seq = torch.FloatTensor(rna_seq).t().squeeze()
                mask = rna_seq >= 0
                print(mask.sum().item())
                rna_seq = rna_seq_feature(rna_seq, mask, is_bulk=True, is_mouse=True)
                xs["bulk_mouse"][name][period] = (rna_seq, torch.BoolTensor(mask))


        for species in self.species[1:]:
            xs[species] = {}
            for name, peca in self.peca_processors[species].raw_peca_data.items():
                print(name, end="\t")
                rna_seq = np.zeros(len(self.gene_names)) - 1
                for index, (gene_name, is_preserve) in enumerate(
                    zip(self.gene_names, self.preserve_gene_mask)
                ):
                    if is_preserve and gene_name in peca.rna_seq.index:
                        rna_seq[index] = peca.rna_seq.loc[gene_name]
                rna_seq = torch.FloatTensor(rna_seq).t().squeeze()
                mask = rna_seq >= 0
                print(mask.sum().item())
                rna_seq = rna_seq_feature(
                    rna_seq,
                    mask,
                    is_bulk="bulk_" in species,
                    is_sc="sc_" in species,
                    is_mouse="mouse" in species,
                    is_human="human" in species,
                )
                xs[species][name] = (rna_seq, torch.BoolTensor(mask))
        return xs
