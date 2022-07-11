# ALIGNNTL: Fine-Tuning

This directory contains information on how to perform fine-tuning using ALIGNN.

### Instructions

The user requires following files in order to start training a model using fine-tuning method
* Sturcture files - contains structure information for a given material (format: `POSCAR`, `.cif`, `.xyz` or `.pdb`) 
* Input-Property file - contains name of the structure file and its corresponding property value (format: `.csv`)
* Configuration file - configuration file with hyperparamters associated with training the model (format: `.json`)
* Pre-trained model - model trained using ALIGNN using any specific materials property (format: `.zip`)

We have provided the an example of Sturcture files (`POSCAR` files), Input-Property file (`id_prop.csv`) and Configuration file (`config_example.json`) in [`examples`](../examples). Download the pre-trained model trained on large datasets from <a href="https://figshare.com/projects/ALIGNN_models/126478">here</a>
