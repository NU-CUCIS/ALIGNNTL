# ALIGNNTL: Feature Extraction

This directory contains information on how to perform feature extraction using ALIGNN.

### Instructions

The user requires following files in order to perform feature extraction
* Sturcture files - contains structure information for a given material (format: `POSCAR`, `.cif`, `.xyz` or `.pdb`) 
* Pre-trained model - model trained using ALIGNN using any specific materials property (format: `.zip`)

We have provided the an example of Sturcture files (`POSCAR` files) in [`examples`](../examples). Download the pre-trained model trained on large datasets from <a href="https://figshare.com/projects/ALIGNN_models/126478">here</a>. 

Now, in order to perform feature extraction, add the details regarding the model in the `all_models` dictionary inside the `train.py` file as described below:
```
all_models = {
    name of the file: [link to the pre-trained model (optional), number of outputs],
    name of the file 2: [link to the pre-trained model 2 (optional), number of outputs],
    ...
    }
```
If the link to the pre-trained model is not provided inside the `all_models` dictionary, place the zip file of the pre-trained model inside the [`alignn`](./alignn) folder. Once the setup for the pre-trained model is done, the feature extraction can be performed by running `create_features.sh` file which contains the follwing code:
```
for filename in ../examples/*.vasp; do
    python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "../examples/data"
done
```
