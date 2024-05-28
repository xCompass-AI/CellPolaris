import os
import random
import time

suffix = random.randint(0, 1000000)
model = "ncf"
# dg_loss_type = "graph_mmd"
# dg_loss_weight = 3e-4
# dg_loss_type = "random_mixup"
dg_loss_weight = 1
dg_loss_type = "graph_mixup"
# dg_loss_weight = 1
# dg_loss_type = "adv_multi"
# dg_loss_weight = 0.1
# fmt: off
# -----------------------
mode = "same_tissue"
keys = ['forebrain', 'midbrain', 'face', 'limb', 'intestine', 'liver', 'kidney', 'lung', 'heart', 'hindbrain', 'neuraltube', 'stomach']
devices = [0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 4, 5]
# ------------------------
# mode = "same_period"
# keys = ["16.5", "15.5", "14.5", "13.5", "12.5", "11.5", "p0", "adult"]
# devices = [7, 6, 5, 4, 3, 2, 1, 0]
# -----------------------
# fmt: on

if __name__ == "__main__":
    for key, device in zip(keys, devices):
        command = (
            f"cd /home/ict/GRNPredict_tr_git_20240524/ && "
            f"nohup /home/ict/anaconda3/envs/ict/bin/python3.10 train_tissue_period_generalization.py "
            f"--mode {mode} "
            f"--select_period_or_tissue {key} "
            f"--device cuda:{device} --model {model} --dg_loss_type {dg_loss_type} "
            f"--dg_loss_weight {dg_loss_weight} "
            # f"--ignore_regression "
            f"--mixup_alpha 0.3 "
            f" >> nohup_tp_{suffix}.out &"
        )
        print(command)
        os.system(command)
        time.sleep(15)
