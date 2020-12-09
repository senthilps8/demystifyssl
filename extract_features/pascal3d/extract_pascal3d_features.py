import hydra.utils as hydra_utils
import hydra
import feature_extractor
import torchvision.transforms as transforms
import os
import torch
import glob
import tempfile
import numpy as np
import scipy.io as sio
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
    listfiles = sorted(
        glob.glob(
            os.path.join(args.dataset.data_dir, 'Image_sets',
                         '*imagenet_train.txt')))
    classes = [os.path.basename(fname).split('_')[0] for fname in listfiles]
    imagenames = {}
    allimagelist = []
    allclasses = []
    allannlist = []
    # Load ImageNet Images
    for cls in classes:
        imagedir = os.path.join(args.dataset.data_dir, 'Images',
                                '{}_imagenet'.format(cls))
        anndir = os.path.join(args.dataset.data_dir, 'Annotations',
                              '{}_imagenet'.format(cls))
        with open(
                os.path.join(args.dataset.data_dir, 'Image_sets',
                             '{}_imagenet_train.txt'.format(cls)), 'r') as f:
            entries = f.read().splitlines()
            imagenames[cls] = entries
        imagelist = [
            os.path.join(imagedir, '{}.JPEG'.format(imname))
            for imname in entries
        ]
        annlist = [(os.path.join(anndir, '{}.mat'.format(imname)), 0)
                   for imname in entries]
        allimagelist.extend(imagelist)
        allclasses.extend([cls for _ in imagelist])
        allannlist.extend(annlist)
        with open(
                os.path.join(args.dataset.data_dir, 'Image_sets',
                             '{}_imagenet_val.txt'.format(cls)), 'r') as f:
            entries = f.read().splitlines()
            imagenames[cls].extend(entries)
        imagelist = [
            os.path.join(imagedir, '{}.JPEG'.format(imname))
            for imname in entries
        ]
        annlist = [(os.path.join(anndir, '{}.mat'.format(imname)), 0)
                   for imname in entries]
        allimagelist.extend(imagelist)
        allclasses.extend([cls for _ in imagelist])
        allannlist.extend(annlist)

    # Load Pascal Images
    for cls in classes:
        imagedir = os.path.join(args.dataset.data_dir, 'Images',
                                '{}_pascal'.format(cls))
        anndir = os.path.join(args.dataset.data_dir, 'Annotations',
                              '{}_pascal'.format(cls))
        with open(
                os.path.join(args.dataset.data_dir, 'Image_sets',
                             '{}_pascal.txt'.format(cls)), 'r') as f:
            fullentries = f.read().splitlines()
            entries = [e.split(' ')[0] for e in fullentries]
            box_ind = [int(e.split(' ')[1]) for e in fullentries]
            imagenames[cls] = entries

        for i in range(len(entries)):
            imname, box_i = entries[i], box_ind[i]
            annfile = os.path.join(anndir, '{}.mat'.format(imname))
            impath = os.path.join(imagedir, '{}.jpg'.format(imname))
            allannlist.append((annfile, box_i))
            allimagelist.append(impath)
            allclasses.append(cls)

    allboxes = []
    for annfile, box_i in allannlist:
        data = sio.loadmat(annfile, struct_as_record=False)
        box = data['record'][0, 0].objects[0, box_i].bbox[0]
        allboxes.append(box)
    allboxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in allboxes]

    print('Found {} images'.format(len(allimagelist)))
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfilelist = tmpfile.name
    with open(tmpfilelist, 'w') as f:
        for fname in allimagelist:
            f.write(fname + '\n')
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpboxlist = tmpfile.name
    with open(tmpboxlist, 'w') as f:
        for box in allboxes:
            f.write('{}\n'.format(','.join([str(b) for b in box])))

    savename = os.path.join(args.dataset.save_dir, 'alldata.pth')
    if not os.path.exists(savename):
        # Load annotations
        allazimuth = []
        allelevation = []
        allsubtypes = []
        for annfile, box_i in allannlist:
            data = sio.loadmat(annfile, struct_as_record=False)
            elev = data['record'][0, 0].objects[0, box_i].viewpoint[
                0, 0].elevation_coarse[0, 0]
            azim = data['record'][0, 0].objects[0, box_i].viewpoint[
                0, 0].azimuth_coarse[0, 0]
            subtype = 'UNK'
            if hasattr(data['record'][0, 0].objects[0, box_i], 'subtype'):
                if len(data['record'][0, 0].objects[0, box_i].subtype) > 0:
                    subtype = data['record'][0, 0].objects[0, box_i].subtype[0]
            allelevation.append(elev)
            allazimuth.append(azim)
            allsubtypes.append(subtype)

        # Extract Features
        args.dataset.filelist = tmpfilelist
        args.dataset.boxlist = tmpboxlist
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        extractor.set_transform(transform)
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
            'azimuth': allazimuth,
            'elevation': allelevation,
            'subtypes': allsubtypes,
            'classes': allclasses,
        }
        torch.save(out, savename)

    ########################################################################
    # Saving Instance change with same viewpoint
    print('Creating Viewpoint Data')
    data = torch.load(savename)
    unique_classes = list(np.unique(data['classes']))
    azimuth = data['azimuth']
    azimuth = [(angle % 180) // 30 for angle in azimuth]
    allinstances = np.zeros(len(azimuth))
    instance_count = np.zeros(6)
    for cls in unique_classes:
        class_ind = np.where(np.array(data['classes']) == cls)[0]
        instance_count = np.ones(6) * np.amax(instance_count)
        for i in class_ind:
            allinstances[i] = instance_count[azimuth[i]]
            instance_count[azimuth[i]] += 1

    data['instance'] = allinstances
    os.makedirs(os.path.dirname(
        savename.replace('pascal3dimnet_', 'pascal3dimnet_viewpointchange_')),
                exist_ok=True)
    torch.save(
        data,
        savename.replace('pascal3dimnet_', 'pascal3dimnet_viewpointchange_'))

    print('Creating Instance Data')
    data = torch.load(savename)
    unique_classes = list(np.unique(data['classes']))
    azimuth = data['azimuth']
    elevation = data['elevation']
    azimuth = [(angle % 180) // 30 for angle in azimuth]
    elevation = [(angle % 90) // 30 for angle in elevation]
    angles = np.array([[azimuth[i], elevation[i]]
                       for i in range(len(elevation))])

    allinstances = np.zeros(len(azimuth))
    instance_count = 0
    for cls in unique_classes:
        class_ind = np.where(np.array(data['classes']) == cls)[0]
        class_angles = angles[class_ind, :]
        uniq_class_angles, rev_index = np.unique(class_angles,
                                                 axis=0,
                                                 return_inverse=True)
        allinstances[class_ind] = rev_index + instance_count
        instance_count += uniq_class_angles.shape[0]

    #############################
    tmpallimagelist = []
    tmpallboxlist = []
    for flist in allimagelist:
        for fname in flist:
            tmpallimagelist.append(fname)
    for boxes in allboxes:
        for box in boxes:
            tmpallboxlist.append(box)
    for k in range(100):
        newallboxes = [allboxes[i] for i in np.where(allinstances == k)[0]]
        newallimages = [
            allimagelist[i] for i in np.where(allinstances == k)[0]
        ]
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfilelist = tmpfile.name
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpboxlist = tmpfile.name
        with open(tmpfilelist, 'w') as f:
            for fname in newallimages:
                f.write(fname + '\n')
        with open(tmpboxlist, 'w') as f:
            for box in newallboxes:
                f.write(','.join([str(b) for b in box]) + '\n')
        args.hparams.batch_size = 10
        args.dataset.filelist = tmpfilelist
        args.dataset.boxlist = tmpboxlist
        extractor.create_dataloader(args)
        feature_dict = extractor.extract_features()

    data['instance'] = allinstances
    os.makedirs(os.path.dirname(
        savename.replace('pascal3dimnet',
                         'pascal3dimnet_viewpointconstant_instance')),
                exist_ok=True)
    torch.save(
        data,
        savename.replace('pascal3dimnet',
                         'pascal3dimnet_viewpointconstant_instance'))
    #######################################################################


if __name__ == '__main__':
    main()
