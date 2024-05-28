from pathlib import Path

import pandas as pd


class SingleCellHumanPECA:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.parent
        self.trs_folder = self.root / "dataset/sc_human/network"
        self.rna_seq_folder = self.root / "dataset/sc_human/RNA/"
        self.trs_files, self.rna_seq_files = self.get_peca_files()

        ## human and mouse
        self.human_to_mouse_file = (
            self.root / "dataset/bulk_human/mouse_gene_to_human_gene.txt"
        )
        self.human_to_mouse_mapping = self.build_human_to_mouse_mapping()
 
        ## human with fly
        # self.human_to_fly_file = (
        #     self.root / "dataset/sc_human/human_fly_one2one.txt"
        # )
        # self.human_to_fly_mapping = self.build_human_to_fly_mapping()
        
        ## human with zebrafish
        # self.human_to_zebrafish_file = (
        #     self.root / "dataset/sc_human/human_zebrafish_one2one.txt"
        # )
        # self.human_to_zebrafish_mapping = self.build_human_to_zebrafish_mapping()

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
            ## human and mouse
            raw_peca_data[name] = RawSingleCellHuman(
                trs_file, self.rna_seq_files[name], self.human_to_mouse_mapping
            )

            # ## human with fly
            # raw_peca_data[name] = RawSingleCellHuman(
            #     trs_file, self.rna_seq_files[name], self.human_to_fly_mapping
            # )

            ## human with zebrafish
            # raw_peca_data[name] = RawSingleCellHuman(
            #     trs_file, self.rna_seq_files[name], self.human_to_zebrafish_mapping
            # )

        return raw_peca_data


    ## human and mouse
    def build_human_to_mouse_mapping(self):
        data = self.human_to_mouse_file.read_text().rstrip("\n").split("\n")[1:]
        mapping = {line[1]: line[0] for line in [line.split("\t") for line in data]}
        return mapping

    ## human with fly
    # def build_human_to_fly_mapping(self):
    #     # human : fly
    #     data = self.human_to_fly_file.read_text().rstrip("\n").split("\n")[1:]
    #     mapping = {line[0]: line[1] for line in [line.split("\t") for line in data]}
    #     return mapping
    
    ## human with zebrafish
    # def build_human_to_zebrafish_mapping(self):
    #     # human : zebrafish
    #     data = self.human_to_zebrafish_file.read_text().rstrip("\n").split("\n")[1:]
    #     mapping = {line[0]: line[1] for line in [line.split("\t") for line in data]}
    #     return mapping


    def __iter__(self):
        for name, raw_peca in self.raw_peca_data.items():
            yield raw_peca


class RawSingleCellHuman:
    ## human and mouse
    def __init__(self, grn_path, rna_seq_path, human_to_mouse):
        self.human_to_mouse = human_to_mouse
    ## human with zebrafish or fly, change 'zebrafish' to 'fly'
    # def __init__(self, grn_path, rna_seq_path, human_to_zebrafish):
    #     self.human_to_zebrafish = human_to_zebrafish

        self.grn_path = grn_path
        self.grn_data = self.load_peca_data()
        self.rna_seq_path = rna_seq_path
        self.rna_seq = self.load_rna_seq_data()
        self.convert_rna_seq()
        self.tf_names, self.tg_names = self.get_tf_and_tg()

    def load_peca_data(self):
        grn_data = pd.read_csv(self.grn_path, sep="\t")
        grn_data = grn_data.iloc[:, :3]
        ## human 2 mouse
        grn_data.iloc[:, 0] = grn_data.iloc[:, 0].apply(
            lambda x: self.human_to_mouse[x] if x in self.human_to_mouse else x
        )
        grn_data.iloc[:, 1] = grn_data.iloc[:, 1].apply(
            lambda x: self.human_to_mouse[x] if x in self.human_to_mouse else x
        )

        ## human with zebrafish or fly, change zebrafish to fly for human with fly
        # grn_data.iloc[:, 0] = grn_data.iloc[:, 0].apply(
        #     lambda x: self.human_to_zebrafish[x] if x in self.human_to_zebrafish else x
        # )
        # grn_data.iloc[:, 1] = grn_data.iloc[:, 1].apply(
        #     lambda x: self.human_to_zebrafish[x] if x in self.human_to_zebrafish else x
        # )
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

    def convert_rna_seq(self):
        ## human 2 mouse
        self.rna_seq.index = self.rna_seq.index.map(
            lambda x: self.human_to_mouse[x] if x in self.human_to_mouse else x
        )

        ## human with zebrafish or fly, change zebrafish to fly for fly
        # self.rna_seq.index = self.rna_seq.index.map(
        #     lambda x: self.human_to_zebrafish[x] if x in self.human_to_zebrafish else x
        # )

        self.rna_seq = self.rna_seq[~self.rna_seq.index.duplicated(keep="first")]



if __name__ == "__main__":
    processor = SingleCellHumanPECA()
    ...
