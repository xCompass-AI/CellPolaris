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
import pandas as pd

warnings.filterwarnings("error")


def load_third_species_link():
    base = Path(__file__).parent / "dataset/third_species"
    zebrafish_path = base / 'TFLink_Danio_rerio_interactions_LS_simpleFormat_v1.0.tsv'
    zebrafish = pd.read_csv(zebrafish_path, sep='\t')
    edges_zebrafish = zebrafish.iloc[:, 4:6]
    fly_path = base / 'TFLink_Drosophila_melanogaster_interactions_LS_simpleFormat_v1.0.tsv'
    fly = pd.read_csv(fly_path, sep='\t')
    edges_fly = fly.iloc[:, 4:6]
    return edges_fly, edges_zebrafish

def to_human_mapping():
    to_human_file = (Path(__file__).parent / "dataset/third_species/human_homologus.txt")
    data = pd.read_csv(to_human_file, sep='\t')
    human_fly = data.iloc[:,[1,3,4]]
    human_fly_nonempty = human_fly.dropna()
    human_fly_one2one = human_fly_nonempty[human_fly_nonempty.iloc[:, -1] == "ortholog_one2one"].iloc[:,:2]

    human_zebrafish = data.iloc[:,[1,7,8]]
    human_zebrafish_nonempty = human_zebrafish.dropna()
    human_zebrafish_one2one = human_zebrafish_nonempty[human_zebrafish_nonempty.iloc[:, -1] == "ortholog_one2one"].iloc[:,:2]

    human_fly_one2one.to_csv('human_fly_one2one.txt', sep='\t', index=False)
    human_zebrafish_one2one.to_csv('human_zebrafish_one2one.txt', sep='\t', index=False)
    return human_fly_one2one, human_zebrafish_one2one

def map_human_edges_2_3spe(homo_one2one, edges_3species, spe):
    edges_filtered = []
    for index, row in edges_3species.iterrows():
        tf = row['Name.TF']
        tg = row['Name.Target']
        if spe == 'fly':
            if tf in homo_one2one['Drosophila melanogaster (Fruit fly) gene name'].values and tg in homo_one2one['Drosophila melanogaster (Fruit fly) gene name'].values:

                edges_filtered.append({'Name.TF': tf, 'Name.Target': tg})
        elif spe == 'zebrafish':
            if tf in homo_one2one['Zebrafish gene name'].values and tg in homo_one2one['Zebrafish gene name'].values:

                edges_filtered.append({'Name.TF': tf, 'Name.Target': tg})
        else:
            print("error_spe")

    edges_filtered_df = pd.DataFrame(edges_filtered)
    edges_filtered_df.reset_index(drop=True, inplace=True)

    return edges_filtered_df


def convert_genename_edges_to_index(edges, species):
    print('begin')
    if species == 'fly':
        human_2spe_genename_2index_path = (Path(__file__).parent / "dataset/third_species/human_2fly_gene_name_to_index.txt")
        edges = map_human_edges_2_3spe(human_fly_one2one, edges_fly, spe='fly')
    elif species == 'zebrafish':
        human_2spe_genename_2index_path = (Path(__file__).parent / "dataset/third_species/human_2zebrafish_gene_name_to_index.txt")
        edges = map_human_edges_2_3spe(human_zebrafish_one2one, edges, spe='zebrafish')

    # mapping to index
    # gene name : index
    human_2spe_genename_2index = pd.read_csv(human_2spe_genename_2index_path, sep="\t")
    mapping_2index = pd.Series(human_2spe_genename_2index['Index'].values, index=human_2spe_genename_2index['GeneName']).to_dict()

    edges.iloc[:, 0] = edges.iloc[:, 0].map(mapping_2index).fillna('Missing')
    edges.iloc[:, 1] = edges.iloc[:, 1].map(mapping_2index).fillna('Missing')

    missing_rows = edges[(edges.iloc[:, 0] == 'Missing') | (edges.iloc[:, 1] == 'Missing')].index
    edges.drop(missing_rows, inplace=True)
    edges.reset_index(drop=True, inplace=True)
    edges.iloc[:, 1] = edges.iloc[:, 1].astype('Int64')
    edges.to_csv(Path(__file__).parent / "dataset/third_species" / f'edges_{species}_index.txt', sep='\t', index=False)


if __name__ == "__main__":
    edges_fly, edges_zebrafish = load_third_species_link()
    human_fly_one2one, human_zebrafish_one2one = to_human_mapping()
    species = 'fly'
    if species == 'fly':
        convert_genename_edges_to_index(edges_fly, 'fly')
    elif species == 'zebrafish':
        convert_genename_edges_to_index(edges_zebrafish, 'zebrafish')
    else:
        print("error")
    print('end')
