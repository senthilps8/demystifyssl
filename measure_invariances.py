import numpy as np
import os
import torch
import hydra
import hydra.utils as hydra_utils


@hydra.main(config_path='configs/feature/', config_name='config')
def main(args):
    args.dataset.data_dir = hydra_utils.to_absolute_path(args.dataset.data_dir)
    args.dataset.save_dir = hydra_utils.to_absolute_path(args.dataset.save_dir)
    args.model.load_path = hydra_utils.to_absolute_path(args.model.load_path)

    method = args.hparams.exp_name
    feat_data = torch.load(os.path.join(args.dataset.save_dir, 'alldata.pth'))
    item_ids = feat_data['instance']
    try:
        # For datasets with class labels
        classes = np.array(feat_data['classes'])
    except BaseException:
        print('WARNING: Using Instance as Class label')
        classes = np.array(feat_data['instance'])

    unique_classes = np.unique(classes)
    if isinstance(item_ids, list):
        item_ids = np.array(item_ids)
    print('Loaded data')

    # 5% is a good max firing rate, so threshold there.
    firing_percent = min((1.0 - 1.0 / len(unique_classes)) * 100, 95)
    print('Setting firing percent to {}'.format(100 - firing_percent))

    print('Computing Thresholds and Global Firing..')

    # Case 1: neuron sign = +1
    feat = feat_data['feat']
    sorted_feat = np.sort(feat, axis=0)
    # Pick value at firing_percent percentile index
    index = int((feat.shape[0] * firing_percent) // 100)
    thresh = sorted_feat[index:index + 1, :]
    firing = feat >= thresh
    meanfiring = np.mean(firing, axis=0, keepdims=True)
    thresholds = thresh
    global_firing = meanfiring

    # Case 1: neuron sign = -1
    feat = feat_data['feat']
    sorted_feat = -1 * np.sort(-1 * feat, axis=0)
    index = int((feat.shape[0] * firing_percent) // 100)
    thresh = sorted_feat[index:index + 1, :]
    firing = feat < thresh
    meanfiring = np.mean(firing, axis=0, keepdims=True)
    neg_thresholds = thresh
    print('Computed Thresholds and Global Firing')

    unique_ids = np.unique(item_ids)
    print('Found {} unique sets in stored data'.format(len(unique_ids)))
    local_firing = [np.zeros_like(global_firing) for cls in unique_classes]
    local_count = [np.zeros_like(global_firing) for cls in unique_classes]
    for itemid in unique_ids:
        ind = np.where(item_ids == itemid)[0]
        class_index = np.where(unique_classes == classes[ind[0]])[0][0]
        if len(ind) < 3:
            continue
        itemfeat = feat_data['feat'][ind]

        # For each neuron only consider the
        # sets where atleast one image fires
        thresh_feat = itemfeat > thresholds
        firing = np.sum(thresh_feat, axis=0, keepdims=True)
        neg_thresh_feat = itemfeat < neg_thresholds
        neg_firing = np.sum(neg_thresh_feat, axis=0, keepdims=True)

        # Choose best sign
        firing = np.maximum(firing, neg_firing)

        pos_ind = firing > 0
        local_firing[class_index][pos_ind] += (firing[pos_ind] /
                                               float(thresh_feat.shape[0]))
        local_count[class_index][pos_ind] += 1

    local_firing = np.concatenate(
        [
            local_firing[ci] / (local_count[ci] + 1e-20)
            for ci in range(len(unique_classes))
        ],
        axis=0,
    )
    print('Computed Local Firing')

    this_inv_scores = np.zeros((6, len(unique_classes)))
    for ci in range(len(unique_classes)):
        score = local_firing[ci] / global_firing[0]
        score = -1 * np.sort(-1 * score)
        cumscore = np.cumsum(score)
        count = np.arange(len(score))
        cummeanscore = cumscore / (count + 1)
        outstr = ''
        for ni, num_neuron in enumerate([10, 25]):
            outstr = outstr + 'Inv Scr {:04d} Neurons: {:.03f},  '.format(
                num_neuron, cummeanscore[num_neuron - 1])
            this_inv_scores[ni, ci] = cummeanscore[num_neuron - 1]

    neuron_counts = [10, 25]
    mean_inv_scores = np.mean(this_inv_scores, axis=1)
    out_str = ' '.join([
        '{:04d}: {:.03f}'.format(neuron_counts[i],
                                 mean_inv_scores[i] * (100 - firing_percent))
        for i in range(len(neuron_counts))
    ])

    print('==============================================================')
    print('Method {:>25},\n{}'.format(method, out_str))
    print('==============================================================')


if __name__ == '__main__':
    main()
