# ALIGNNTL: Fine-Tuning

This directory contains information on how to perform fine-tuning using ALIGNN.

### Instructions

The user requires following files in order to start training a model using fine-tuning method
* Sturcture files - contains structure information for a given material (format: `POSCAR`, `.cif`, `.xyz` or `.pdb`) 
* Input-Property file - contains name of the structure file and its corresponding property value (format: `.csv`)
* Configuration file - configuration file with hyperparamters associated with training the model (format: `.json`)
* Pre-trained model - model trained using ALIGNN using any specific materials property (format: `.zip`)

We have provided the an example of Sturcture files (`POSCAR` files), Input-Property file (`id_prop.csv`) and Configuration file (`config_example.json`) in [`examples`](../examples). Download the pre-trained model trained on large datasets from <a href="https://figshare.com/projects/ALIGNN_models/126478">here</a>. 

Now, in order to perform fine-tuning based transfer learning, add the details regarding the model in the `all_models` dictionary inside the `train.py` file as described below:
```
all_models = {
    name of the file: [link to the pre-trained model (optional), number of outputs],
    name of the file 2: [link to the pre-trained model 2 (optional), number of outputs],
    ...
    }
```
If the link to the pre-trained model is not provided inside the `all_models` dictionary, place the zip file of the pre-trained model inside the [`alignn`](./alignn) folder. Once the setup for the pre-trained model is done, the model training can be performed as follows:
```
python alignn/train_folder.py --root_dir "../examples" --config "../examples/config_example.json" --id_prop_file "id_prop.csv" --output_dir=model
```
Make sure that the Input-Property file `--id_prop_file` is placed inside the root directory `--root_dir` where Sturcture files are present.
