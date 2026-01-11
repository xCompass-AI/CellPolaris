# CellPolaris

## Abstract

Cell fate decisions are orchestrated by intricate gene regulatory networks (GRNs), which govern gene expression with precise spatiotemporal control. However, accurately capturing context-specific nature of gene regulation remains challenging, particularly when integrating multi-omics data at bulk and single-cell level across diverse cellular contexts.

Here, we present CellPolaris, a unified computational framework designed to decode the roles of transcription factors (TFs) in developmental processes. CellPolaris performs TF-centered GRN construction, master TF identification, and TF perturbation simulation. By leveraging transfer learning, the framework generates tissue-specific or cell-type-specific GRNs using pre-constructed high-confidence GRNs of diverse contexts and requires only transcriptomic data as input. Using these learned GRNs, CellPolaris identifies underlying master TFs critical for cell fate transitions and simulates the effects of TF perturbations on developmental processes. Benchmarking tests demonstrate the robust performance of CellPolaris in GRN construction. The efficacy of CellPolaris is supported by the significant overlap between predicted top-ranked master regulators and known TF combinations experimentally validated in cell fate conversion experiments. Furthermore, CellPolaris accurately simulates the developmental consequences of Rfx2 knockout during round spermatid differentiation. In summary, we present CellPolaris, a comprehensive framework that enables GRN construction through transfer learning, identification of key TFs driving cell fate transitions, and simulation of TF perturbations. This tool allows us to further elucidate the regulatory mechanisms underlying developmental processes and cell state transitions.


## Workflow


![image](https://github.com/xCompass-AI/CellPolaris/assets/49229942/c35a6212-5fee-4488-b3c4-5e53c7035d71)
Figure 1. Schematic overview of CellPolaris design. 
a Generation of a generalized transfer model using PECA2 tool to construct a GRN database from ATAC-Seq and corresponding cell state RNA-Seq data. This model enables cross-species and cross-tissue analysis. The trained model can take RNA-Seq data as input and generate corresponding GRNs, which can be utilized for downstream applications. 
b Prediction of cell fate regulatory factors based on GRN analysis: Comparison of GRN differences between source and target cells during cell fate transitions. Transcription factor nodes in the differential networks were scored and ranked. 
c Simulation of the impact of gene perturbation on cell differentiation pathways based on GRN analysis: A probability graphical model of gene expression regulation was constructed using GRN analysis. Simulated knockout genes were set to zero, and the expression changes of neighboring node genes were inferred to deduce changes in cell states.

## 1 Use transfer_learning to get GRN:

### Dataset
To download the dataset required to run the transfer learning code from the link below:
https://www.scidb.cn/en/s/VNvY3e

### Dependency

```bash
conda env create -f ./transfer_learning/environment.yml
```

### Running
For cross-tissue/period
```bash
python ./transfer_learning/scripts_for_execute/script_train_tissue_period.py
```

For cross-cluster
```bash
python ./transfer_learning/scripts_for_execute/script_train_multi_to_multi.py
```
Then modify the relevant parameters (e.g., species, fold, etc.) and run the code below to generate the GRN
```bash
python ./transfer_learning/scripts_for_execute/generate_grn_multi_to_multi_generalization.py
```

For cross-species
```bash
python ./transfer_learning/scripts_for_execute/script_train_cross_species_extramixup.py
```
then generate GRN by
```bash
python ./transfer_learning/scripts_for_execute/script_generate_cross_species_extramixup.py
```


## 2 Use model_PGM to get deltaX:

### Dependency
Creating an Environment from a YAML File
```bash
conda env create -f environment_pgm.yml
```
### Running
```bash
python ./model_PGM/run.py
```

## 3 Use plot to get final results:
### Running
```bash
python ./plot/CellCruise.py
```

# Troubleshooting
If you encounter any problems during installation or use of CellPolaris, please contact us by email cuiwentao@cnic.cn. We will help you as soon as possible.

# Citation
If you use this code for your research, please cite our paper [CellPolaris: Transfer Learning for Gene Regulatory Network Construction to Guide Cell State Transitions](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202508697)
```bash
Feng, G., Qin, X., Zhang, J., Huang, W., Zhang, Y., Cui, W., ... & Li, X. CellPolaris: Transfer Learning for Gene Regulatory Network Construction to Guide Cell State Transitions. Advanced science (Weinheim, Baden-Wurttemberg, Germany), e08697.
```
