import torch 

def get_dataset_name(mode):
    datasets = {
        "ade20k": "Ade20kDataset",
        "cityscapes": "CityscapesDataset",
        "coco": "CocoStuffDataset",
        "bosch": "BoschDataset",
        "aoi": "AoiDataset",
        "celeba": "CelebAHQDataset",
    }

    if not mode in datasets:
        raise ValueError("There is no such dataset regime as %s" % mode)

    return datasets[mode]


def get_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)
    print(dataset_name)

    file = __import__("oasis.dataloaders." + dataset_name).__dict__["dataloaders"]
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](
        opt, for_metrics=False
    )
    dataset_val = file.__dict__[dataset_name].__dict__[dataset_name](
        opt, for_metrics=True
    )
    print(
        "Created %s, size train: %d, size val: %d"
        % (dataset_name, len(dataset_train), len(dataset_val))
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    return dataloader_train, dataloader_val
