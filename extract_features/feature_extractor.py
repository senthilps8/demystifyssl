import hydra
import utils
import torchvision.models as models
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


@torch.no_grad()
def extract_features_from_dataloader(model, image_loader):
    feature_dict = {}
    for i, (images, filenames, boxes) in enumerate(
            tqdm(image_loader, leave=False, desc='ImageLoader')):
        images = images.cuda(non_blocking=True)
        features = model(images).cpu().numpy()

        for bi in range(len(images)):
            feature_dict[filenames[bi]] = features[bi]
    return feature_dict


class FeatureExtractor:
    def __init__(self, args):
        self.create_model(args)
        self.create_transform(args)
        self.create_dataloader(args)

    def create_model(self, args):
        model = models.__dict__[args.model.arch](
            num_classes=args.model.feat_dim,
            pretrained=args.model.imnet_pretrained)
        if args.model.saved_fc_type == 'none':
            model.fc = nn.Sequential()
        elif args.model.saved_fc_type == 'mlp':
            dim_mlp = model.fc.weight.shape[1]
            model.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                model.fc,
            )

        if args.model.load_path and os.path.isfile(args.model.load_path):
            checkpoint = torch.load(args.model.load_path)
            state_dict = checkpoint['state_dict']
            if args.model.replace_key:
                tmp = {}
                for k in state_dict.keys():
                    newk = k
                    for pair in args.model.replace_key:
                        newk = newk.replace(pair[0], pair[1])
                    newk = newk.replace('module.', '')
                    tmp[newk] = state_dict[k]
                state_dict = tmp
            print('Missing keys: {}'.format(
                set(model.state_dict().keys()) - set(state_dict.keys())))
            model.load_state_dict(state_dict, strict=False)

        if args.model.extract_before_fc:
            model.fc = nn.Sequential()
        model = model.eval()
        model = torch.nn.parallel.DataParallel(model).cuda()
        self.model = model

    def create_transform(self, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        finalsize = args.dataset.inputsize
        self.transform = transforms.Compose([
            transforms.Resize(finalsize),
            transforms.ToTensor(),
            normalize,
        ])

    def set_transform(self, transform):
        self.transform = transform

    def create_dataloader(self, args):
        self.image_loader = torch.utils.data.DataLoader(
            utils.ListDataset(args.dataset.filelist,
                              boxlist=args.dataset.boxlist,
                              transform=self.transform,
                              polygon=args.dataset.polygon),
            batch_size=args.hparams.batch_size,
            shuffle=False,
            num_workers=args.hparams.workers,
            pin_memory=True,
        )

    def extract_features(self):
        feature_dict = extract_features_from_dataloader(
            self.model, self.image_loader)
        return feature_dict


@hydra.main(config_path='./configs/feature/config.yaml')
def main(args):
    extractor = FeatureExtractor(args)
    feature_dict = extractor.extract_features(args)
    torch.save(feature_dict, args.hparams.save_path)


if __name__ == '__main__':
    main()
