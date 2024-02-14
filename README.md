# ALIGNNTL

This repository contains the code for ALIGNN-based transfer learning framework to predict materials properties using structure-based inputs. The code provides the following functions:

* Train a ALIGNN model on a given dataset.
* Use a pre-trained ALIGNN model to perform fine-tuning based transfer learning on a given target dataset.
* Use a pre-trained ALIGNN model to perform feature extraction on a given target dataset.
* Predict material properties of new compounds with a pre-trained ALIGNN model.

## Installation Requirements

The basic requirement for using the files are a Python 3.8 with the packages listed in `setup.py`. It is advisable to create an virtual environment with the correct dependencies. Please refer to the guidelines <a href="https://github.com/usnistgov/alignn">here</a> for installation details.

The work related experiments was performed on Linux Fedora 7.9 Maipo. The code should be able to work on other Operating Systems as well but it has not been tested elsewhere.

## Source Files
  
Here is a brief description about the folder content:

* [`FineTuning`](./FineTuning): code to perform fine-tuning based transfer learning.

* [`FeatureExtraction`](./FeatureExtraction): code to perform feature extraction.

* [`example`](./example): example dataset to perform fine-tuning or feature extraction.

## ALIGNNTL: Fine-Tuning

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
If the link to the pre-trained model is not provided inside the `all_models` dictionary, place the zip file of the pre-trained model inside the [`alignn`](./alignn) folder. Once the setup for the pre-trained model is done, the fine-tuning based model training can be performed as follows:
```
python alignn/train_folder.py --root_dir "../examples" --config "../examples/config_example.json" --id_prop_file "id_prop.csv" --output_dir=model
```
Make sure that the Input-Property file `--id_prop_file` is placed inside the root directory `--root_dir` where Sturcture files are present.

## ALIGNNTL: Feature Extraction

The user requires following files in order to perform feature extraction
* Sturcture files - contains structure information for a given material (format: `POSCAR`, `.cif`, `.xyz` or `.pdb`) 
* Input-Property file - contains name of the structure file and its corresponding property value (format: `.csv`)
* Pre-trained model - model trained using ALIGNN using any specific materials property (format: `.zip`)

We have provided the an example of Sturcture files (`POSCAR` files) and Input-Property file (`id_prop.csv`) in [`examples`](../examples). Download the pre-trained model trained on large datasets from <a href="https://figshare.com/projects/ALIGNN_models/126478">here</a>. 

Now, in order to perform feature extraction, add the details regarding the model in the `all_models` dictionary inside the `train.py` file as described below:
```
all_models = {
    name of the file: [link to the pre-trained model (optional), number of outputs],
    name of the file 2: [link to the pre-trained model 2 (optional), number of outputs],
    ...
    }
```
If the link to the pre-trained model is not provided inside the `all_models` dictionary, place the zip file of the pre-trained model inside the [`alignn`](./alignn) folder. Once the setup for the pre-trained model is done, the feature extraction can be performed by running the `create_features.sh` script file which contains the following code:
```
for filename in ../examples/*.vasp; do
    python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "../examples/data"
done
```
The script will convert the structure files into atom (x), bond (y) and angle (z) based features one-by-one (batch-wise conversion has not been implemented yet).  Example: `abc.vasp` will produce `abc_x.csv` (9 atom-based features), `abc_y.csv` (9 bond-based features) and `abc_z.csv` (5 angle-based features). 

Once you have converted all the structure files in the Input-Property file `id_prop.csv` using the script file, run the jupyter notebooks `pre-processing.ipynb` to convert the structure-wise features into a dataset. Pre-processing steps contained within the `pre-processing.ipynb` file is as follows:
* Attach the appropriate property value and identifier (jid) to each of the extracted features file based on id_prop.csv 
* Create a seperate file for each of the features (atom, bond, angle) based on the extracted checkpoints
* Create combined features (in the order of atom, bond and angle) from same (3-1) or different (3-2) checkpoints. Use first 512 features for atom+bond and all features for atom+bon+angle as input for model training.
* (Optional) Divide each of the files into train, validation and test files based on the json file `ids_train_val_test.json` available in the output directory of the ALIGNN model

## Using Pre-Trained Model
All the trained models are available at [Zenodo](https://doi.org/10.5072/zenodo.1222572), and these models can be used to make predictions directly.

To perform prediction using the ALIGNN model, please refer to https://github.com/usnistgov/alignn

To perform prediction using the ElemNet model, please refer to https://github.com/NU-CUCIS/CrossPropertyTL

## Developer Team

The code was developed by Vishu Gupta from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Publication

1. Vishu Gupta, Kamal Choudhary,  Brian DeCost, Francesca Tavazza, Carelyn Campbell, Wei-keng Liao, Alok Choudhary, and Ankit Agrawal, “Structure-aware graph neural network based deep transfer learning framework for enhanced predictive analytics on diverse materials datasets,” npj Computational Materials 10.1 (2024): 1. [<a href="https://www.nature.com/articles/s41524-023-01185-3">DOI</a>] [<a href="https://www.nature.com/articles/s41524-023-01185-3.pdf">PDF</a>]

```tex
@article{gupta2024structure,
  title={Structure-aware graph neural network based deep transfer learning framework for enhanced predictive analytics on diverse materials datasets},
  author={Gupta, Vishu and Choudhary, Kamal and DeCost, Brian and Tavazza, Francesca and Campbell, Carelyn and Liao, Wei-keng and Choudhary, Alok and Agrawal, Ankit},
  journal={npj Computational Materials},
  volume={10},
  number={1},
  pages={1},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgements

The open-source implementation of ALIGNN <a href="https://github.com/usnistgov/alignn">here</a> provided significant initial inspiration for the structure of this code base.

## Disclaimer

The research code shared in this repository is shared without any support or guarantee of its quality. However, please do raise an issue if you find anything wrong, and I will try my best to address it.

email: vishugupta2020@u.northwestern.edu

Copyright (C) 2023, Northwestern University.

See COPYRIGHT notice in top-level directory.

## Funding Support

This work was performed under the following financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from NSF award CMMI-2053929, and DOE awards DE-SC0019358, DE-SC0021399, and Northwestern Center for Nanocombinatorics.
