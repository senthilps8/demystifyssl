MODELPATH=checkpoints/moco_v2_200ep_pretrain.pth.tar
MODELNAME=moco
LOAD_PARAMS="model.replace_key=[['module.encoder_q.','']] model.saved_fc_type='mlp' model.feat_dim=128"

# ALOI Viewpoint
python extract_features/ALOI/extract_generic_ALOI_features.py dataset=aloi dataset.type=viewpoint model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}
python measure_invariances.py dataset=aloi dataset.type=viewpoint model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}

# ALOI Illumination Color
python extract_features/ALOI/extract_generic_ALOI_features.py dataset=aloi dataset.type=illumcolor model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}
python measure_invariances.py dataset=aloi dataset.type=illumcolor model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}

# ALOI Illumination Direction
python extract_features/ALOI/extract_generic_ALOI_features.py dataset=aloi dataset.type=illumdir model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}
python measure_invariances.py dataset=aloi dataset.type=illumdir model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}

# GOT-10K
python extract_features/GOT/extract_GOT10k_features.py dataset=got model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}
python measure_invariances.py dataset=got model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}

# Pascal3D
python extract_features/pascal3d/extract_pascal3d_features.py dataset=pascal3d model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}
python measure_invariances.py dataset=pascal3d dataset.name=pascal3dimnet_viewpointconstant_instance model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}
python measure_invariances.py dataset=pascal3d dataset.name=pascal3dimnet_viewpointchange model.load_path=$MODELPATH hparams.exp_name=$MODELNAME ${LOAD_PARAMS}

