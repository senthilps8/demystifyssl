import hydra.utils as hydra_utils
import hydra
import feature_extractor
import os
import torch
import glob
import tempfile
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(config_path='../../configs/feature/config.yaml')
def main(args):
    args.dataset.data_dir = hydra_utils.to_absolute_path(args.dataset.data_dir)
    args.dataset.save_dir = hydra_utils.to_absolute_path(args.dataset.save_dir)
    args.model.load_path = hydra_utils.to_absolute_path(args.model.load_path)
    if os.path.exists(os.path.join(args.dataset.save_dir, 'alldata.pth')):
        print('Found {}'.format(
            os.path.join(args.dataset.save_dir, 'alldata.pth')))
        sys.exit(0)

    extractor = feature_extractor.FeatureExtractor(args)
    allimagelist = []
    allinstances = []

    # ALOI 1000 instances
    for i in range(1, 1001):
        dname = os.path.join(args.dataset.data_dir, '{}'.format(i))
        filenames = sorted(glob.glob(os.path.join(dname, '*')))
        for fname in filenames:
            allimagelist.append(fname)
            allinstances.append(i)

    print('Found {} images'.format(len(allimagelist)))
    # Write to temp files in expected format
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfilelist = tmpfile.name
    with open(tmpfilelist, 'w') as f:
        for fname in allimagelist:
            f.write(fname + '\n')

    savename = os.path.join(args.dataset.save_dir, 'alldata.pth')
    if not os.path.exists(savename):
        print('File does not exist: {}'.format(savename))
        # Extract Features
        args.dataset.filelist = tmpfilelist
        extractor.create_dataloader(args)
        os.makedirs(args.dataset.save_dir, exist_ok=True)
        feature_dict = extractor.extract_features()
        feat_size = len(feature_dict[allimagelist[0]])
        num_feat = len(allimagelist)
        catfeat = np.zeros((num_feat, feat_size))
        for fi, fname in enumerate(allimagelist):
            catfeat[fi, :] = feature_dict[fname]

        out = {
            'feat': catfeat,
            'instance': allinstances,
            'classes': allinstances,
        }
        torch.save(out, savename)


if __name__ == '__main__':
    main()
