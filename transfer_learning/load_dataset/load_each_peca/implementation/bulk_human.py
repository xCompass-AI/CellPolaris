from collections import defaultdict
from pathlib import Path

import pandas as pd


class BulkHumanPECA:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.parent
        self.trs_folder = self.root / "dataset/bulk_human/Networks"
        self.rna_seq_file = self.root / "dataset/bulk_human/humanRNA/HumanRNA.txt"
        self.trs_files = self.get_peca_files()
        self.human_to_mouse_file = (
            self.root / "dataset/bulk_human/mouse_gene_to_human_gene.txt"
        )
        self.human_to_mouse_mapping = self.build_human_to_mouse_mapping()
        self.raw_peca_data = self.load_raw_peca()

    def get_peca_files(self):
        trs_files = {}
        for fp in self.trs_folder.iterdir():
            if "network" in fp.stem:
                trs_files["_".join(fp.stem.split("_")[:-1])] = fp
        return trs_files

    def load_raw_peca(self):
        raw_peca_data = {}
        all_rna_seq = pd.read_csv(self.rna_seq_file, sep="\t", index_col=0, header=0)
        for name, trs_file in self.trs_files.items():
            rna_seq = all_rna_seq.loc[:, name]
            rna_seq = rna_seq[~rna_seq.index.duplicated(keep="first")]
            raw_peca_data[name] = RawBulkHuman(
                trs_file, rna_seq, self.human_to_mouse_mapping
            )
        return raw_peca_data

    def build_human_to_mouse_mapping(self):
        data = self.human_to_mouse_file.read_text().rstrip("\n").split("\n")[1:]
        mapping = {line[1]: line[0] for line in [line.split("\t") for line in data]}
        return mapping

    def __iter__(self):
        for name, raw_peca in self.raw_peca_data.items():
            yield raw_peca


class RawBulkHuman:
    def __init__(self, grn_path, rna_seq, human_to_mouse):
        self.human_to_mouse = human_to_mouse
        self.grn_path = grn_path
        self.grn_data = self.load_peca_data()
        self.rna_seq = rna_seq
        self.convert_rna_seq()
        self.tf_names, self.tg_names = self.get_tf_and_tg()

    def load_peca_data(self):
        grn_data = pd.read_csv(self.grn_path, sep="\t")
        grn_data = grn_data.iloc[:, :3]

        grn_data.iloc[:, 0] = grn_data.iloc[:, 0].apply(
            lambda x: self.human_to_mouse[x] if x in self.human_to_mouse else x
        )
        grn_data.iloc[:, 1] = grn_data.iloc[:, 1].apply(
            lambda x: self.human_to_mouse[x] if x in self.human_to_mouse else x
        )
        return grn_data

    def get_tf_and_tg(self):
        tf_names = self.grn_data.iloc[:, 0].unique()
        tg_names = self.grn_data.iloc[:, 1].unique()
        return tf_names, tg_names

    def convert_rna_seq(self):
        self.rna_seq.index = self.rna_seq.index.map(
            lambda x: self.human_to_mouse[x] if x in self.human_to_mouse else x
        )


if __name__ == "__main__":
    processor = BulkHumanPECA()
    ...
