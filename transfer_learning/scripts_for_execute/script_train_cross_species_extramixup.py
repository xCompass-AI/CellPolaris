import os
import random
import time

suffix = random.randint(0, 1000000)
species = "bulk_mouse bulk_human"
model = "ncf"
dg_loss_type = "graph_extramixup"
dg_loss_weight = 1
top_ratio = 0.5

targets = ["heart"]
tissue_list = "heart kidney spleen liver lung"

devices = [0]

if __name__ == "__main__":
    for target, device in zip(targets, devices):
        command = (
            f"cd /home/ict/GRNPredict_tr_git_20240524/ && "
            f"nohup /home/ict/anaconda3/envs/ict_q/bin/python train_cross_species_generalization_multisource_mouse2human.py "
            f"--device cuda:{device} --species {species} --model {model} --target {target} --tissue_list {tissue_list} "
            f"--dg_loss_type {dg_loss_type} "
            f"--dg_loss_weight {dg_loss_weight} "
            f"--mixup_alpha 0.5 "
            f"--top_ratio {top_ratio} "
            f"--batch_size 1280 "
            f"--epoch 100 "
            f" >> nohup_mt2mt_{suffix}.out &"
        )
        print(command)
        os.system(command)
        time.sleep(10)
