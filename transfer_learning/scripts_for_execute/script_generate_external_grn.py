import os
from pathlib import Path

external_rna_seq_folder = Path(
    "/home/ict/GRNPredict_tr_git_20240524/dataset/downstream_test/mouse_jingzi_RS2_8/pseudobulk"
)

rna_seqs = list(external_rna_seq_folder.glob("*.txt"))
os.system(
    f"cd /home/ict/GRNPredict_tr_git_20240524/ && ~/anaconda3/envs/ict/bin/python3.10  generate_external_grn.py --species sc_mouse --fold 0 --transfer_loss_type graph_mixup --model ncf --seed 0 --device cuda:0 --rna_seq_path {' '.join([str(rna_seq) for rna_seq in rna_seqs])} "
)
