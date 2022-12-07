import random
import torch
import torchvision
from torch.nn import functional as F
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np


class CocoStuffDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        if opt.phase == "test" or for_metrics:
            opt.load_size = 256
        else:
            opt.load_size = 286
        opt.crop_size = 256
        opt.label_nc = 182
        opt.contain_dontcare_label = True
        opt.semantic_nc = 183  
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(
        self,
    ):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert("RGB")
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255 + 1
        label[label == 256] = 0 
        return {"image": image, "label": label, "name": self.images[idx]}

    def get_single_item(self, name):
        img_path = "/fs/scratch/rng_cr_bcai_dl/OpenData/cocostuff/val_img/"
        lbl_path = "/fs/scratch/rng_cr_bcai_dl/OpenData/cocostuff/val_label/"

        img_path = os.path.join(img_path, f"{name}.jpg")
        lbl_path = os.path.join(lbl_path, f"{name}.png")

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)
        self.opt.no_flip = True
        image, label = self.transforms(image, label)
        label = label * 255 + 1
        label[label == 256] = 0  
        return {"image": image, "label": label, "name": name}

    def get_single_label(self, name):
        lbl_path = "/fs/scratch/rng_cr_bcai_dl/OpenData/cocostuff/val_label/"
        lbl_path = os.path.join(lbl_path, f"{name}.png")
        label = Image.open(lbl_path)
        self.opt.no_flip = True

        label = self.label_transforms(label)
        label = label * 255 + 1
        label[label == 256] = 0  
        return {"label": label, "name": name}

    def get_single_label_fast(self, name):
        lbl_path = "/fs/scratch/rng_cr_bcai_dl/OpenData/cocostuff/val_label/"
        lbl_path = os.path.join(lbl_path, f"{name}.png")
        
        label = torchvision.io.read_image(lbl_path)
        label = label.unsqueeze(0)
        
        self.opt.no_flip = True
        label = self.label_transforms_fast(label, noflip=True)
        
        
        label = label + 1
        label[label == 256] = 0  
        
        return {"label": label, "name": name}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        path_img = os.path.join(self.opt.dataroot, mode + "_img")
        path_lab = os.path.join(self.opt.dataroot, mode + "_label")
        images = sorted(os.listdir(path_img))
        labels = sorted(os.listdir(path_lab))
        assert len(images) == len(
            labels
        ), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert (
                os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0]
            ), "%s and %s are not matching" % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop(
            (crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size)
        )
        label = label.crop(
            (crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size)
        )
        
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label

    def label_transforms_fast(self, label, noflip=False):
        
        
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        label = F.interpolate(label, size=(new_width, new_height), mode="nearest")
        
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        label = label[
            :,
            :,
            crop_x : crop_x + self.opt.crop_size,
            crop_y : crop_y + self.opt.crop_size,
        ]
        
        if not (
            self.opt.phase == "test" or self.opt.no_flip or self.for_metrics or noflip
        ):
            if random.random() < 0.5:
                label = TR.functional.hflip(label)

        return label

    def label_transforms(self, label):
        
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))

        label = label.crop(
            (crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size)
        )
        
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                label = TR.functional.hflip(label)
        
        label = TR.functional.to_tensor(label)
        return label
