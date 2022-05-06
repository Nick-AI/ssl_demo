""" Functions to load dataset from ImageFolder.

Supports generic classification datafolder structure:
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png
Code adapted from https://github.com/HobbitLong/PyContrast.
"""

import torch
from torchvision import datasets


class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """

    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None, return_labels=False):
        super(ImageFolderInstance, self).__init__(
            root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()
        self.return_labels = return_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            if self.return_labels:
                return img, index, jigsaw_image, target
            return img, index, jigsaw_image
        else:
            if self.return_labels:
                return img, index, target
            return img, index
