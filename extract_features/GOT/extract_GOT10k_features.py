import hydra.utils as hydra_utils
import hydra
import feature_extractor
import os
import torch
import tempfile
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(config_path='../../configs/feature/config.yaml')
def main(args):
    args.dataset.data_dir = hydra_utils.to_absolute_path(args.dataset.data_dir)
    args.dataset.save_dir = hydra_utils.to_absolute_path(args.dataset.save_dir)
    args.model.load_path = hydra_utils.to_absolute_path(args.model.load_path)
    print(args.pretty())
    # Set seed so same samples are chosen
    np.random.seed(1992)
    # If data exists, save time and skip this
    if os.path.exists(os.path.join(args.dataset.save_dir, 'alldata.pth')):
        print('Found {}'.format(
            os.path.join(args.dataset.save_dir, 'alldata.pth')))
        sys.exit(0)

    extractor = feature_extractor.FeatureExtractor(args)
    # Frame level attributes
    attributes = ['absence', 'cover']
    with open(os.path.join(args.dataset.data_dir, 'list.txt'), 'r') as f:
        vids = f.read().splitlines()

    # Too many videos?
    if len(vids) > args.dataset.max_videos:
        vids = vids[::len(vids) // args.dataset.max_videos]

    allimagelist = []
    allboxes = []
    allann = []
    allframeids = []
    vid_to_class = {}
    for vid in tqdm(vids, desc='Reading GOT Annotations'):
        # Load video metadata
        metafile = os.path.join(args.dataset.data_dir, vid, 'meta_info.ini')
        with open(metafile, 'r') as f:
            data = f.read().splitlines()[1:]
            data = [
                d.split(': ')[1].replace(' ', '_') for d in data
                if 'object_class' in d
            ]
        vid_to_class[vid] = data[0]

        # Read bounding boxes
        boxfile = os.path.join(args.dataset.data_dir, vid, 'groundtruth.txt')
        with open(boxfile, 'r') as f:
            boxtext = f.read()
        boxes = boxtext.splitlines()[::args.dataset.stride]
        allboxes.append(boxes)

        num_boxes = len(boxtext.splitlines())
        fname_fmt = os.path.join(args.dataset.data_dir, vid, '{:08d}.jpg')
        imagelist = [fname_fmt.format(i + 1) for i in range(num_boxes)]
        imagelist = imagelist[::args.dataset.stride]
        allimagelist.append(imagelist)

        # Load absence and occlusion annotations
        thisann = []
        for attr in attributes:
            ann_file = os.path.join(args.dataset.data_dir, vid,
                                    attr + '.label')
            if not os.path.exists(ann_file):
                ann = [-1] * num_boxes
            else:
                with open(ann_file, 'r') as f:
                    ann = list(map(int, f.read().splitlines()))
            thisann.append(np.array(ann[::args.dataset.stride]))
        allann.append(thisann)
        allframeids.append(np.arange(num_boxes)[::args.dataset.stride])
    print('Found {} images'.format(
        sum([len(imlist) for imlist in allimagelist])))

    # Create temp file list to extract features
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfilelist = tmpfile.name
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpboxlist = tmpfile.name
    with open(tmpfilelist, 'w') as f:
        for flist in allimagelist:
            for fname in flist:
                f.write(fname + '\n')
    with open(tmpboxlist, 'w') as f:
        for boxes in allboxes:
            for box in boxes:
                f.write(box + '\n')

    # Extract Features
    if not os.path.exists(os.path.join(args.dataset.save_dir,
                                       vids[0] + '.pth')):
        args.dataset.filelist = tmpfilelist
        args.dataset.boxlist = tmpboxlist
        extractor.create_dataloader(args)
        os.makedirs(args.dataset.save_dir, exist_ok=True)
        feature_dict = extractor.extract_features()

        feat_size = len(feature_dict[allimagelist[0][0]])
        # Save features for each video
        for ci, vid in enumerate(tqdm(vids, desc='Saving features')):
            num_feat = len(allimagelist[ci])
            vidfeat = np.zeros((num_feat, feat_size))
            for fi, fname in enumerate(allimagelist[ci]):
                vidfeat[fi, :] = feature_dict[fname]

            boxes = allboxes[ci]
            boxes = [list(map(float, box.split(','))) for box in boxes]
            boxes = np.array(boxes)

            out = {
                'feat': vidfeat,
                'boxes': boxes,
                'ids': allframeids[ci],
                'class': vid_to_class[vid]
            }
            attribute_ann = {
                attributes[i]: allann[ci][i]
                for i in range(len(attributes))
            }
            out.update(attribute_ann)
            savename = os.path.join(args.dataset.save_dir, vid + '.pth')
            torch.save(out, savename)

    # Merge features for good videos (heuristic - see paper)
    features = []
    instances = []
    allcovers = []
    allclasses = []
    count = 0
    for ci, vid in enumerate(tqdm(vids, desc='Accumulating features')):
        savename = os.path.join(args.dataset.save_dir, vid + '.pth')
        data = torch.load(savename)
        feat = data['feat']
        cover = data['cover']
        uniq_covers = np.unique(cover)
        if uniq_covers.max() - uniq_covers.min() < 4:
            continue
        chosen_inds = []
        for c in uniq_covers:
            chosen_inds.append(np.random.choice(np.where(cover == c)[0]))

        allcovers.extend(list(uniq_covers))
        feat = feat[np.array(chosen_inds)]
        features.append(feat)
        instances.extend([count] * len(chosen_inds))
        allclasses.extend([data['class']] * len(chosen_inds))
        count += 1

    features = np.vstack(features)
    torch.save(
        {
            'feat': features,
            'instance': instances,
            'cover': allcovers,
            'classes': allclasses
        }, os.path.join(args.dataset.save_dir, 'alldata.pth'))


if __name__ == '__main__':
    main()
