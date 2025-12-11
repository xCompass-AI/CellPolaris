# CellPolaris

## Abstract

Changes in cell fate are determined by a gene regulatory network (GRN), a sophisticated system that regulates gene expression with precise spatiotemporal patterns. However, accurately capturing causal and context specific gene regulations by integrating multi-omics data at bulk and single cell level remains challenging. In this study, we introduce CellPolaris, a computational system that leverages transfer learning algorithms to generate tissue or cell-type-specific GRNs. The system employs high-confidence GRNs generated from collected transcriptomic data and paired ATAC-Seq data for training. Once trained, the model only requires transcriptomic data as input to infer the GRNs of the corresponding states. Benchmarking tests demonstrated the reliability of CellPolaris in GRN construction. Notably, to assess the model's scalability, we incorporated GRNs from other species such as mouse and zebrafish, which slightly improved the performance of CellPolaris in predicting GRNs for human cells.  Subsequently, based on the GRNs, CellPolaris can predict master transcription factors (TFs) involved in cell fate transitions and simulate the impact of TF perturbations on developmental processes. The results show that the top-ranked master regulatory factors identified by CellPolaris largely overlap with the combinations of factors that have successfully achieved cell fate conversion in selected reprogramming experimental systems. Simultaneously, CellPolaris predicts the changes in target genes (TGs) with in silico knockout of TFs and then simulates the effects of these knockouts on round spermatid differentiation. In summary, CellPolaris provides an approach for utilizing transcriptomic data to construct GRNs by transfer learning, thereby identifying potential TFs involved in cell fate transitions and deciphering the underlying mechanisms of developmental regulation.


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
If you use this code for your research, please cite our paper [CellPolaris: Decoding Cell Fate through Generalization Transfer Learning of Gene Regulatory Networks](https://biorxiv.org/content/10.1101/2023.09.25.559244v1.abstract)
```bash
@article{feng2023cellpolaris,
  title={CellPolaris: Decoding Cell Fate through Generalization Transfer Learning of Gene Regulatory Networks},
  author={Feng, Guihai and Qin, Xin and Zhang, Jiahao and Huang, Wuliang and Zhang, Yiyang and Cui, Wentao and Li, Shirui and Chen, Yao and Liu, Wenhao and Tian, Yao and others},
  journal={bioRxiv},
  pages={2023--09},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
