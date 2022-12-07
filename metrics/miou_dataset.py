import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def imresize(im, size, interp="bilinear"):
    if interp == "nearest":
        resample = Image.NEAREST
    elif interp == "bilinear":
        resample = Image.BILINEAR
    elif interp == "bicubic":
        resample = Image.BICUBIC
    else:
        raise Exception("resample method undefined!")

    return im.resize(size, resample)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        self.padding_constant = opt.padding_constant
        self.parse_input_list(odgt, **kwargs)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, "r")]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print("

    def img_transform(self, img):
        
        img = np.float32(np.array(img)) / 255.0
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        self.batch_record_list = [[], []]

        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            
            this_sample = self.list_sample[self.cur_idx]
            if this_sample["height"] > this_sample["width"]:
                self.batch_record_list[0].append(this_sample)  
            else:
                self.batch_record_list[1].append(this_sample)  

            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        batch_records = self._get_sub_batch()

        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = (
                batch_records[i]["height"],
                batch_records[i]["width"],
            )
            this_scale = min(
                this_short_size / min(img_height, img_width),
                self.imgMaxSize / max(img_height, img_width),
            )
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(
            self.round2nearest_multiple(batch_width, self.padding_constant)
        )
        batch_height = int(
            self.round2nearest_multiple(batch_height, self.padding_constant)
        )

        assert (
            self.padding_constant >= self.segm_downsampling_rate
        ), "padding constant must be equal or large than segm downsamping rate"
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate,
        ).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            image_path = os.path.join(self.root_dataset, this_record["fpath_img"])
            segm_path = os.path.join(self.root_dataset, this_record["fpath_segm"])

            img = Image.open(image_path).convert("RGB")
            segm = Image.open(segm_path)
            assert segm.mode == "L"
            assert img.size[0] == segm.size[0]
            assert img.size[1] == segm.size[1]

            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            img = imresize(img, (batch_widths[i], batch_heights[i]), interp="bilinear")
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp="nearest")

            segm_rounded_width = self.round2nearest_multiple(
                segm.size[0], self.segm_downsampling_rate
            )
            segm_rounded_height = self.round2nearest_multiple(
                segm.size[1], self.segm_downsampling_rate
            )
            segm_rounded = Image.new("L", (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (
                    segm_rounded.size[0] // self.segm_downsampling_rate,
                    segm_rounded.size[1] // self.segm_downsampling_rate,
                ),
                interp="nearest",
            )

            img = self.img_transform(img)

            segm = self.segm_transform(segm)

            batch_images[i][:, : img.shape[1], : img.shape[2]] = img
            batch_segms[i][: segm.shape[0], : segm.shape[1]] = segm

        output = dict()
        output["img_data"] = batch_images
        output["seg_label"] = batch_segms
        return output

    def __len__(self):
        return int(
            1e10
        )  
        
class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        image_path = os.path.join(self.root_dataset, this_record["fpath_img"])
        segm_path = os.path.join(self.root_dataset, this_record["fpath_segm"])
        img = Image.open(image_path).convert("RGB")
        segm = Image.open(segm_path).convert("L")
        assert segm.mode == "L"
        
        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            
            scale = min(
                this_short_size / float(min(ori_height, ori_width)),
                self.imgMaxSize / float(max(ori_height, ori_width)),
            )
            target_height, target_width = int(ori_height * scale), int(
                ori_width * scale
            )

            target_width = self.round2nearest_multiple(
                target_width, self.padding_constant
            )
            target_height = self.round2nearest_multiple(
                target_height, self.padding_constant
            )

            img_resized = imresize(
                img, (target_width, target_height), interp="bilinear"
            )

            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output["img_ori"] = np.array(img)
        output["img_data"] = [x.contiguous() for x in img_resized_list]
        output["seg_label"] = batch_segms.contiguous()
        output["info"] = this_record["fpath_img"]
        
        return output

    def __len__(self):
        return self.num_sample

class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        
        image_path = this_record["fpath_img"]
        img = Image.open(image_path).convert("RGB")

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            
            scale = min(
                this_short_size / float(min(ori_height, ori_width)),
                self.imgMaxSize / float(max(ori_height, ori_width)),
            )
            target_height, target_width = int(ori_height * scale), int(
                ori_width * scale
            )

            target_width = self.round2nearest_multiple(
                target_width, self.padding_constant
            )
            target_height = self.round2nearest_multiple(
                target_height, self.padding_constant
            )

            img_resized = imresize(
                img, (target_width, target_height), interp="bilinear"
            )

            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output["img_ori"] = np.array(img)
        output["img_data"] = [x.contiguous() for x in img_resized_list]
        output["info"] = this_record["fpath_img"]
        return output

    def __len__(self):
        return self.num_sample
