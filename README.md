# ALIGNNTL

This repository contains the code for ALIGNN-based transfer learning framework to predict materials properties using structure-based inputs. The code provides the following functions:

* Train a ALIGNN model on a given dataset.
* Use a pre-trained ALIGNN model to perform fine-tuning based transfer learning on a given target dataset.
* Use a pre-trained ALIGNN model to perform feature extraction on a given target dataset.
* Predict material properties of new compounds with a pre-trained ALIGNN model.

## Installation Requirements

The basic requirement for using the files are a Python 3.8 with the packages listed in `setup.py`. It is advisable to create an virtual environment with the correct dependencies. Please refer to the guidelines <a href="https://github.com/usnistgov/alignn">here</a> for installation details

The work related experiments was performed on Linux Fedora 7.9 Maipo. The code should be able to work on other Operating Systems as well but it has not been tested elsewhere.

## Source Files
  
Here is a brief description about the folder content:

* [`FineTuning`](./FineTuning): code to perform fine-tuning based transfer learning

* [`FeatureExtraction`](./FeatureExtraction): code to perform feature extraction

* [`example`](./example): example dataset to perform fine-tuning or feature extraction

## Developer Team

The code was developed by Vishu Gupta from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Publication

## Acknowledgements

The open-source implementation of ALIGNN <a href="https://github.com/usnistgov/alignn">here</a> provided significant initial inspiration for the structure of this code-base.

## Disclaimer

The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you find anything wrong and I will try my best to address it.

email: vishugupta2020@u.northwestern.edu

Copyright (C) 2021, Northwestern University.

See COPYRIGHT notice in top-level directory.

## Funding Support

This work was performed under the following financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from NSF award CMMI-2053929, and DOE awards DE-SC0019358, DE-SC0021399, and Northwestern Center for Nanocombinatorics.
