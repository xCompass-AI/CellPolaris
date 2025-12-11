from pathlib import Path

import pandas as pd


class SingleCellMousePECA:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.parent
        self.trs_folder = self.root / "dataset/sc_mouse/network"
        self.rna_seq_folder = self.root / "dataset/sc_mouse/RNA/"
        self.trs_files, self.rna_seq_files = self.get_peca_files()
        self.raw_peca_data = self.load_raw_peca()

    def get_peca_files(self):
        trs_files, rna_seq_files = {}, {}
        for fp in self.trs_folder.iterdir():
            if "network" in fp.stem:
                file_name = "_".join(fp.stem.split("_")[:-1])
                trs_files[file_name] = fp
                rna_seq_files[file_name] = self.rna_seq_folder / f"{file_name}.txt"
        return trs_files, rna_seq_files

    def load_raw_peca(self):
        raw_peca_data = {}
        for name, trs_file in self.trs_files.items():
            raw_peca_data[name] = RawSingleCellMouse(trs_file, self.rna_seq_files[name])
        return raw_peca_data

    def __iter__(self):
        for name, raw_peca in self.raw_peca_data.items():
            yield raw_peca


class RawSingleCellMouse:
    def __init__(self, grn_path, rna_seq_path):
        self.grn_path = grn_path
        self.grn_data = self.load_peca_data()
        self.rna_seq_path = rna_seq_path
        self.rna_seq = self.load_rna_seq_data()
        self.tf_names, self.tg_names = self.get_tf_and_tg()

    def load_peca_data(self):
        grn_data = pd.read_csv(self.grn_path, sep="\t")
        grn_data = grn_data.iloc[:, :3]
        return grn_data

    def load_rna_seq_data(self):
        rna_seq_data = pd.read_csv(
            self.rna_seq_path, sep="\t", header=None, index_col=0
        )
        rna_seq_data = rna_seq_data[~rna_seq_data.index.duplicated(keep="first")]
        return rna_seq_data.squeeze()

    def get_tf_and_tg(self):
        tf_names = self.grn_data.iloc[:, 0].unique()
        tg_names = self.grn_data.iloc[:, 1].unique()
        return tf_names, tg_names


if __name__ == "__main__":
    processor = SingleCellMousePECA()
    ...
