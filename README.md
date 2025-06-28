# MDJCL-DDI

## Requirement
```
python=3.7
torch=1.13.0
torch_geometric=2.3.1
pandas=1.3.5
rdkit=2022.09.1
cython=3.0.12
subword_nmt=0.3.8
```
When installing these packages, other unlisted packages will be installed.


## Usage 
```
  usage: train_binary.py [--device]
                         [--savemodels_dir]
                         [--dataset_path]
                         [--ESPF_path]
                         [--train_3D_structure]
                         [--val_3D_structure]
                         [--test_3D_structure]
```

## Data Specification
All training, validation, test should follow specification to be parsed correctly by MDJCL-DDI.
The
```
data/
├── train.csv
├── val.csv
├── test.csv
├── sdf/
│    └── All drugs' sdf file from drugbank
└── word_segmentation/
     └── NULL
```
```
ESPF/
├── drug_codes_chembl.txt
└── subword_units_map_chembl.csv
```

## Data Preprocessing
The process of data preprocessing has been integrated into the train_binary program. When this program is run, the preprocessed content will be automatically saved in the dataset folder.

## run

The following presents the running commands of the model in the dataset of Pang et al

`python train_binary.py --device cuda:0 --savemodels_dir ckpt/ dataset_path ./dataset/ --ESPF_path ./ESPF/ --train_3D_structure drug_3D_structure_train.pkl --val_3D_structure drug_3D_structure_val.pkl --test_3D_structure drug_3D_structure_test.pkl`
