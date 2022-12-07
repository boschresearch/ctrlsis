import random
import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np

class Ade20kDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        if opt.phase == "test" or for_metrics:
            opt.load_size = 256
        else:
            opt.load_size = 286
        opt.crop_size = 256
        opt.label_nc = 150
        opt.contain_dontcare_label = True
        opt.semantic_nc = 151  
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics

        if "image_list_train" in opt.__dict__:
            self.images, self.labels, self.paths = self.list_images(
                opt.image_list_train, opt.image_list_val
            )
        else:
            self.images, self.labels, self.paths = self.list_images()

    def __len__(
        self,
    ):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert("RGB")
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}

    def get_single_item(self, name):
        if "train" in name:
            img_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/images/training/"
            lbl_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/annotations/training/"
        elif "val" in name:
            img_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/images/validation/"
            lbl_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/annotations/validation/"
        img_path = os.path.join(img_path, f"{name}.jpg")
        lbl_path = os.path.join(lbl_path, f"{name}.png")
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)
        image, label = self.transforms(image, label, noflip=True)
        label = label * 255
        return {"image": image, "label": label, "name": name}

    def get_single_label(self, name):
        if "train" in name:
            lbl_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/annotations/training/"
        elif "val" in name:
            lbl_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/annotations/validation/"
        lbl_path = os.path.join(lbl_path, f"{name}.png")
        label = Image.open(lbl_path)
        label = self.label_transforms(label, noflip=True)
        label = label * 255
        return {"label": label, "name": name}

    def get_single_label_fast(self, name):
        if "train" in name:
            lbl_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/annotations/training/"
        elif "val" in name:
            lbl_path = "/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADEChallengeData2016/annotations/validation/"
        lbl_path = os.path.join(lbl_path, f"{name}.png")
        label = torchvision.io.read_image(lbl_path)
        label = label.unsqueeze(0)
        label = self.label_transforms_fast(label, noflip=True)
        
        return {"label": label, "name": name}

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

    def list_images(self, image_list_train=None, image_list_val=None):

        mode = (
            "validation" if self.opt.phase == "test" or self.for_metrics else "training"
        )
        path_img = os.path.join(self.opt.dataroot, "images", mode)
        path_lab = os.path.join(self.opt.dataroot, "annotations", mode)
        img_list = os.listdir(path_img)
        lab_list = os.listdir(path_lab)
        img_list = [
            filename
            for filename in img_list
            if ".png" in filename or ".jpg" in filename
        ]
        lab_list = [
            filename
            for filename in lab_list
            if ".png" in filename or ".jpg" in filename
        ]

        if mode == "training" and image_list_train is not None:
            img_list = [
                im
                for im in img_list
                if im.replace(".jpg", "").replace(".png", "") in image_list_train
            ]
            lab_list = [
                im
                for im in lab_list
                if im.replace(".jpg", "").replace(".png", "") in image_list_train
            ]
        elif mode == "validation" and image_list_val is not None:
            img_list = [
                im
                for im in img_list
                if im.replace(".jpg", "").replace(".png", "") in image_list_val
            ]
            lab_list = [
                im
                for im in lab_list
                if im.replace(".jpg", "").replace(".png", "") in image_list_val
            ]
        images = sorted(img_list)
        labels = sorted(lab_list)
        print("len(images) =", len(images))
        print("len(labels) =", len(labels))

        assert len(images) == len(
            labels
        ), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert (
                os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0]
            ), "%s and %s are not matching" % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label, noflip=False):
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
        
        if not (
            self.opt.phase == "test" or self.opt.no_flip or self.for_metrics or noflip
        ):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label

    def label_transforms(self, label, noflip=False):
        
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        label = label.crop(
            (crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size)
        )
        
        if not (
            self.opt.phase == "test" or self.opt.no_flip or self.for_metrics or noflip
        ):
            if random.random() < 0.5:
                label = TR.functional.hflip(label)
        
        label = TR.functional.to_tensor(label)
        return label
