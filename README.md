# CellPolaris

## Abstract

Cell fate changes are determined by gene regulatory network (GRN), a sophisticated system regulating gene expression in precise spatial and temporal patterns. However, existing methods for reconstructing GRNs suffer from inherent limitations, leading to compromised accuracy and application generalizability. In this study, we introduce CellPolaris, a computational system that leverages transfer learning algorithms to generate high-quality, cell-type-specific GRNs. Diverging from conventional GRN inference models, which heavily rely on integrating epigenomic data with transcriptomic information or adopt causal strategies through gene co-expression networks, CellPolaris employs high-confidence GRN sources for model training, relying exclusively on transcriptomic data to generate previously unknown cell-type-specific GRNs. Applications of CellPolaris demonstrate remarkable efficacy in predicting master regulatory factors and simulating in-silico perturbations of transcription factors during cell fate transition, attaining state-of-the-art performance in accurately predicting candidate key factors and outcomes in cell reprogramming and spermatogenesis with validated datasets. It is worth noting that, with a transfer learning framework, CellPolaris can perform GRN based predictions in all cell types even across species. Together, CellPolaris represents a significant advancement in deciphering the mechanisms of cell fate regulation, thereby enhancing the precision and efficiency of cell fate manipulation at high resolution.

## Workflow


![image](https://github.com/xCompass-AI/CellPolaris/assets/49229942/7f27d271-a674-406b-95a0-9142a0ae4cf8)

Figure 1. The schematic overview of CellPolaris design. 
(A) Generation of a generalized transfer model using PECA2 tool to construct a GRN database from ATAC-Seq and corresponding cell state RNA-Seq data. This model enables cross-species and cross-tissue analysis. The trained model can take RNA-Seq data as input and generate corresponding GRNs, which can be utilized for downstream applications. 
(B) Prediction of cell fate regulatory factors based on GRN analysis: Comparison of GRN differences between source and target cells during cell fate transitions. Transcription factor nodes in the differential networks were scored and ranked.
(C) Simulation of the impact of gene perturbation on cell differentiation pathways based on GRN analysis: A probability graphical model of gene expression regulation was constructed using GRN analysis. Simulated knockout genes were set to zero, and the expression changes of neighboring node genes were inferred to deduce changes in cell states.
## Dependency


Creating an Environment from a YAML File
```bash
conda env create -f environment.yml
```
## Dataset
To download the dataset required to run the transfer learning code from the link below:
https://1drv.ms/f/s!AiR3trb0uGdXgqAzIOCs4zrBVYOqew?e=FmGoBW

## Running
Use model_PGM to get deltaX through command:

```bash
python ./model_PGM/run.py
```

Use plot to get final results
