import random
import torch
import torchvision
from torchvision import transforms as TR
import os
from PIL import Image
from torch.nn import functional as F

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35  # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

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
        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}

    def get_single_label_fast(self, name, val=True):
        truncated_name = "_".join(name.split("_")[:3])
        if val:
            lbl_path = "/fs/scratch/rng_cr_bcai_dl/OpenData/cityscapes/gtFine/val/"
        else:
            lbl_path = "/fs/scratch/rng_cr_bcai_dl/OpenData/cityscapes/gtFine/train/"

        lbl_path = os.path.join(lbl_path, f"{truncated_name}_gtFine_labelIds.png")
        label = torchvision.io.read_image(lbl_path)
        label = label.unsqueeze(0)

        label = self.label_transforms_fast(label, noflip=True)
        return {"label": label, "name": name}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        sub_folders = os.listdir(path_lab)
        sub_folders = [
            s for s in sub_folders if os.path.isdir(os.path.join(path_lab, s))
        ]
        for city_folder in sorted(sub_folders):
            cur_folder = os.path.join(path_lab, city_folder)

            subsub_folder = os.listdir(cur_folder)
            for item in sorted(subsub_folder):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))

        assert len(images) == len(
            labels
        ), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace(
                "_gtFine_labelIds.png", ""
            ), "%s and %s are not matching" % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (
            int(self.opt.load_size / self.opt.aspect_ratio),
            self.opt.load_size,
        )
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label

    def label_transforms_fast(self, label, noflip=False):
        # label is already tensor
        # resize
        new_width, new_height = (
            int(self.opt.load_size / self.opt.aspect_ratio),
            self.opt.load_size,
        )

        label = F.interpolate(label, size=(new_width, new_height), mode="nearest")

        # flip
        if not (
            self.opt.phase == "test" or self.opt.no_flip or self.for_metrics or noflip
        ):
            if random.random() < 0.5:
                label = TR.functional.hflip(label)

        return label
