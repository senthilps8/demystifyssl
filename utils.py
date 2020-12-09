from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
from torchvision.transforms import functional as transF
import torch.utils.data as data


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ListDataset(data.Dataset):
    def __init__(self, filelist, boxlist=None, transform=None, polygon=False):
        self.transform = transform
        self.filenames = []
        if filelist:
            with open(filelist, 'r') as f:
                self.filenames = f.read().splitlines()

        self.boxes = None
        if boxlist:
            with open(boxlist, 'r') as f:
                boxes = f.read().splitlines()
                self.boxes = np.array(
                    [list(map(float, b.split(','))) for b in boxes])
                if polygon:
                    tmpboxes = []
                    for box in self.boxes:
                        if len(box) == 4:
                            tmpboxes.append(box)
                        else:
                            tmpboxes.append([
                                min(box[::2]),
                                min(box[1::2]),
                                max(box[::2]) - min(box[::2]),
                                max(box[1::2]) - min(box[1::2])
                            ])
                    self.boxes = np.array(tmpboxes)
                self.boxes = self.boxes[:, [1, 0, 3, 2]]

        assert (self.boxes is None or len(self.filenames) == len(self.boxes))

    def __getitem__(self, index):
        img = pil_loader(self.filenames[index])
        box = torch.empty(0)
        if self.boxes is not None:
            box = self.boxes[index]
            img = transF.crop(img, *box)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.filenames[index], box

    def __len__(self):
        return len(self.filenames)
