import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


class BulkMousePECA:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.parent
        self.trs_files, self.rna_seq_files = self.get_peca_files()
        self.raw_peca_data = self.load_raw_peca()

    def get_peca_files(self):
        trs_folder = self.root / "dataset/bulk_mouse/TRS"
        trs_files = defaultdict(dict)
        for fp in trs_folder.iterdir():
            if "Adult" in fp.name:
                if fp.stem == "EncodeA20BalbcannMAdult8wks_network":
                    trs_files["A20Balbcann".lower()]["adult"] = fp
                else:
                    tissue_name = fp.name.split("Encode")[1].split("C57bl6")[0]
                    trs_files[tissue_name.lower()]["adult"] = fp
            elif "P0" in fp.name:
                tissue_name = fp.name.split("P0")[0]
                trs_files[tissue_name.lower()]["p0"] = fp
            else:
                tissue_name = fp.name.split("E")[0]
                time_point = fp.name.split("E")[1].split("_")[0]
                trs_files[tissue_name.lower()][time_point] = fp
        rna_seq_folder = self.root / "dataset/bulk_mouse/RNAseq"
        rna_seq_files = defaultdict(dict)
        regrex = re.compile(r"([a-zA-Z]+)([0-9.]+)RM1.txt")
        for fp in rna_seq_folder.iterdir():
            if fp.name == 'cross_species_generated_grn':
                continue
            if "Adult" in fp.name:
                if fp.stem == "EncodeA20BalbcannMAdult8wks":
                    rna_seq_files["A20Balbcann".lower()]["adult"] = fp
                else:
                    tissue_name = fp.name.split("Encode")[1].split("C57bl6")[0]
                    rna_seq_files[tissue_name.lower()]["adult"] = fp
            elif "0" in fp.name:
                tissue_name = fp.name.split("0")[0]
                rna_seq_files[tissue_name.lower()]["p0"] = fp
            else:
                tissue_name, time_point = regrex.findall(fp.name)[0]
                rna_seq_files[tissue_name.lower()][time_point] = fp

        return trs_files, rna_seq_files

    def load_raw_peca(self):
        raw_peca_data = defaultdict(dict)
        for tissue, periods in self.trs_files.items():
            for period, trs_file in periods.items():
                rna_seq_file = self.rna_seq_files[tissue][period]
                raw_peca_data[tissue][period] = RawBulkMouse(trs_file, rna_seq_file)
        return raw_peca_data

    def __iter__(self):
        for tissue, periods in self.raw_peca_data.items():
            for period, raw_peca in periods.items():
                yield raw_peca


class RawBulkMouse:
    def __init__(self, grn_path, rna_seq_path):
        self.grn_path = grn_path
        self.rna_seq_path = rna_seq_path
        self.grn_data = self.load_peca_data()
        self.rna_seq = self.load_rna_seq_data()
        self.tf_names, self.tg_names = self.get_tf_and_tg()

    def load_peca_data(self):
        grn_data = pd.read_csv(self.grn_path, sep="\t")
        grn_data = grn_data.iloc[:, :3]
        grn_data.iloc[1:, 2] = grn_data.iloc[1:, 2].astype("float")
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
    processor = BulkMousePECA()
    ...
