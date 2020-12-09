# Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases

This repository includes code to accompany the paper: 

```
@article{purushwalkam2020demystifying,
  title={Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases},
  author={Purushwalkam Shiva Prakash, Senthil and Gupta, Abhinav},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Datasets 

In the paper, we present invariances measured using three datasets - [ALOI](https://aloi.science.uva.nl/) for measuring viewpoint, illumination and color invariances, [Pascal3D](https://cvgl.stanford.edu/projects/pascal3d.html) for instance and viewpoint+instance invariances, and [GOT10K](http://got-10k.aitestunion.com/) for occlusion invariance. The first step is to download these datasets (or the datasets in which you're interested). Download the datasets to `data/` in the code folder. 

#### ALOI Dataset
The ALOI dataset can be downloaded to the right format by running these commands in the code folder:

```
mkdir data
cd data
wget http://aloi.science.uva.nl/tars/aloi_red2_view.tar
wget http://aloi.science.uva.nl/tars/aloi_red2_col.tar
wget http://aloi.science.uva.nl/tars/aloi_red2_ill.tar
tar -xf aloi_red2_view.tar
mv png2 ALOI_viewpoint
tar -xf aloi_red2_col.tar
mv png2 ALOI_illumcolor
tar -xf aloi_red2_ill.tar
mv png2 ALOI_illumdir
```

#### Pascal3D Dataset
Similarly, the Pascal3D dataset can be downloaded to the right format by running these commands in the code folder:

```
mkdir data
cd data
wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
unzip PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1 pascal3d
```

#### GOT-10K Dataset
The GOT-10K dataset requires registration. So please download the dataset from http://got-10k.aitestunion.com/. Place the folder containing the training set at `data/GOT/`. So the structure of the data folder would look like:
```
data/
└── GOT -> ../GOT
    ├── GOT-10k_Train_000001
    │   ├── 00000001.jpg
    │   ├── 00000002.jpg
    │   ├── 00000003.jpg
    │   ├── ............
    │   ├── 00000119.jpg
    │   ├── 00000120.jpg
    │   ├── absence.label
    │   ├── cover.label
    │   ├── cut_by_image.label
    │   ├── groundtruth.txt
    │   └── meta_info.ini
    ├── GOT-10k_Train_000002
    │   ├── 00000001.jpg
    │   ├── 00000002.jpg
    │   ├── 00000003.jpg
    │   ├── ............
    │   ├── 00000068.jpg
    │   ├── 00000069.jpg
    │   ├── 00000070.jpg
    │   ├── absence.label
    │   ├── cover.label
    │   ├── cut_by_image.label
    │   ├── groundtruth.txt
    │   └── meta_info.ini
    ├── ....................
    ├── ....................
    └── GOT-10k_Train_009335
        ├── 00000001.jpg
        ├── 00000002.jpg
        ├── 00000003.jpg
        ├── ............
        ├── 00000089.jpg
        ├── 00000090.jpg
        ├── 00000091.jpg
        ├── absence.label
        ├── cover.label
        ├── cut_by_image.label
        ├── groundtruth.txt
        └── meta_info.ini
```

## Environment Setup
Install `pytorch` and `torchvision` from https://pytorch.org/.
Install other dependencies:
```
pip install hydra-core --upgrade

pip install -e git+https://github.com/tqdm/tqdm.git@master#egg=tqdm
# OR 
# conda install -c conda-forge tqdm

```

## Measuring Invariances

For a model saved at `$MODELPATH`, we can extract features and measure the corresponding invariances using:
```
# GOT-10K
python extract_features/GOT/extract_GOT10k_features.py dataset=got model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
python measure_invariances.py dataset=got model.load_path=$MODELPATH hparams.exp_name=$MODELNAME

# Pascal3D
python extract_features/pascal3d/extract_pascal3d_features.py dataset=pascal3d model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
python measure_invariances.py dataset=pascal3d dataset.name=pascal3dimnet_viewpointconstant_instance model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
python measure_invariances.py dataset=pascal3d dataset.name=pascal3dimnet_viewpointchange model.load_path=$MODELPATH hparams.exp_name=$MODELNAME

# ALOI Viewpoint
python extract_features/ALOI/extract_generic_ALOI_features.py dataset=aloi type=viewpoint model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
python measure_invariances.py dataset=aloi type=viewpoint model.load_path=$MODELPATH hparams.exp_name=$MODELNAME

# ALOI Illumination Color
python extract_features/ALOI/extract_generic_ALOI_features.py dataset=aloi type=illumcolor model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
python measure_invariances.py dataset=aloi type=illumcolor model.load_path=$MODELPATH hparams.exp_name=$MODELNAME

# ALOI Illumination Direction
python extract_features/ALOI/extract_generic_ALOI_features.py dataset=aloi type=illumdir model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
python measure_invariances.py dataset=aloi type=illumdir model.load_path=$MODELPATH hparams.exp_name=$MODELNAME
```
where `$MODELNAME` can be set to any arbitrary name to identify the model (used to name the file where features are saved). In order to simplify loading of a variety of models, we provide additional configuration parameters `model.replace_key`, `model.saved_fc_type` and `model.feat_dim` to accommodate model files with different parameter key names. See the code and examples below for details.


## Reproducing Results

#### ImageNet Supervised model
Run `./experiments/supervised.sh`.

#### Moco V2
Download the Moco V2 checkpoint model from https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar to `checkpoints/`. Run `./experiments/mocov2.sh` to estimate the invariances.

#### PIRL
Download the PIRL model from https://drive.google.com/file/d/1cegLn9p2Z75N7DfNNmDGAej0mvvLYfWL/view?usp=sharing to `checkpoints/`. Note: this is our best reproduction of PIRL and not the official model. Run `./experiments/pirl.sh` to estimate the invariances.



