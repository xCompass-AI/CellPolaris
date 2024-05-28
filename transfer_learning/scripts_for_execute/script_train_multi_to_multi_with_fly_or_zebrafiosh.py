import os
import random
import time

suffix = random.randint(0, 1000000)
species = "sc_human"
# species = "sc_mouse"
# species = "sc_mouse sc_human"
# species = "bulk_human bulk_mouse"
# species = "bulk_mouse"
# species = "bulk_human"
model = "ncf"
# dg_loss_type = "None"
dg_loss_type = "graph_mixup"
# dg_loss_type = "graph_mmd"
# dg_loss_weight = 3e-4
# dg_loss_type = "random_mixup"
# dg_loss_type = "adv_multi"
dg_loss_weight = 1
add_spe3 = 'fly'
top_ratio = 0.5

if __name__ == "__main__":
    for index in range(0, 4):
        command = (
            f"cd /home/ict/GRNPredict_tr_git_20240524/ && "
            f"nohup /home/ict/anaconda3/envs/ict/bin/python3.10 train_multi_to_multi_with_fly_or_zebrafish.py "
            f"--fold {index} --device cuda:{index} --species {species} --model {model} "
            f"--dg_loss_type {dg_loss_type} "

            f"--dg_loss_weight {dg_loss_weight} "
            # f"--append_bulk "
            # f"--ignore_regression "
            f"--mixup_alpha 0.5 "
            f"--top_ratio {top_ratio} "
            f"--batch_size 1280 "
            f"--epoch 100 "
            f"--add_spe3 {add_spe3}"
            f" >> nohup_mt2mt_{suffix}.out &"
        )
        print(command)
        os.system(command)
        time.sleep(10)
