"""This module loads the several supported pre-trained GAN Models on given datasets. 
"""
import torch
from types import SimpleNamespace
from oasis.models import models


def _load_opt_oasis_cityscapes() -> SimpleNamespace:
    opt = SimpleNamespace(
        EMA_decay=0.9999,
        add_vgg_loss=False,
        batch_size=4,
        beta1=0.0,
        beta2=0.999,
        channels_D=64,
        channels_G=64,
        checkpoints_dir="/fs/scratch/rng_cr_bcai_dl/dave2rng/cityscapes_final/",
        continue_train=True,
        dataroot="/fs/scratch/rng_cr_bcai_dl/OpenData/cityscapes/",
        dataset_mode="cityscapes",
        freq_fid=5000,
        freq_print=1000,
        freq_save_ckpt=20000,
        freq_save_latest=10000,
        freq_save_loss=2500,
        freq_smooth_loss=250,
        gpu_ids="0",
        lambda_labelmix=10.0,
        lambda_vgg=10.0,
        loaded_latest_iter=126199,
        lr_d=0.0004,
        lr_g=0.0001,
        name="20201224_city_main",
        no_3dnoise=False,
        no_EMA=False,
        no_balancing_inloss=False,
        no_flip=False,
        no_labelmix=False,
        no_spectral_norm=False,
        num_epochs=200,
        num_res_blocks=6,
        param_free_norm="syncbatch",
        phase="test",
        eed=42,
        spade_ks=3,
        which_iter="best",
        z_dim=64,
        crop_size=512,
        load_size=512,
        aspect_ratio=2.0,
        img_channels=3,
        label_nc=34,
        contain_dontcare_label=True,
        semantic_nc=35,
        cache_filelist_read=False,
        cache_filelist_write=False,
        lambda_bneck=0,
        num_workers=4,
        ckpt_iter="best",
    )
    return opt


specific_parameters = {
    "coco": {
        "semantic_nc": 183,
        "aspect_ratio": 1.0,
        "name": "20201225_coco_main",
        "dataset_mode": "coco",
        "EMA_decay": 0.9999,
        "add_vgg_loss": False,
        "batch_size": 32,
        "beta1": 0.0,
        "beta2": 0.999,
        "channels_D": 64,
        "channels_G": 64,
        "checkpoints_dir": "/fs/scratch/rng_cr_bcai_dl/dave2rng/esc2rng_exp/",
        "continue_train": True,
        "dataroot": "/fs/scratch/rng_cr_bcai_dl/OpenData/cocostuff/",
        "dataset_mode": "coco",
        "freq_fid": 5000,
        "freq_print": 1000,
        "freq_save_ckpt": 20000,
        "freq_save_latest": 10000,
        "freq_save_loss": 2500,
        "freq_smooth_loss": 250,
        "gpu_ids": 0,
        "lambda_labelmix": 10.0,
        "lambda_vgg": 10.0,
        "loaded_latest_iter": 360000,
        "lr_d": 0.0004,
        "lr_g": 0.0001,
        "no_3dnoise": False,
        "no_EMA": False,
        "no_balancing_inloss": False,
        "no_flip": False,
        "no_labelmix": False,
        "no_spectral_norm": False,
        "num_epochs": 100,
        "num_res_blocks": 6,
        "param_free_norm": "syncbatch",
        "phase": "train",
        "seed": 42,
        "spade_ks": 3,
        "which_iter": "best",
        "z_dim": 64,
    },
}


def _load_opt_oasis_coco() -> SimpleNamespace:
    opt = SimpleNamespace(
        EMA_decay=0.9999,
        add_vgg_loss=False,
        batch_size=1,
        beta1=0.0,
        beta2=0.999,
        channels_D=64,
        channels_G=64,
        checkpoints_dir="/fs/scratch/rng_cr_bcai_dl/dave2rng/esc2rng_exp/",
        continue_train=True,
        dataroot="/fs/scratch/rng_cr_bcai_dl/OpenData/cocostuff/",
        dataset_mode="coco",
        freq_fid=5000,
        freq_print=1000,
        freq_save_ckpt=20000,
        freq_save_latest=10000,
        freq_save_loss=2500,
        freq_smooth_loss=250,
        gpu_ids=0,
        lambda_labelmix=10.0,
        lambda_vgg=10.0,
        loaded_latest_iter=360000,
        lr_d=0.0004,
        lr_g=0.0001,
        name="20201225_coco_main",
        no_3dnoise=False,
        no_EMA=False,
        no_balancing_inloss=False,
        no_flip=False,
        no_labelmix=False,
        no_spectral_norm=False,
        num_epochs=200,
        num_res_blocks=6,
        param_free_norm="syncbatch",
        phase="test",
        seed=42,
        spade_ks=3,
        which_iter="best",
        z_dim=64,
        crop_size=256,
        aspect_ratio=1.0,
        img_channels=3,
        semantic_nc=183,
        lambda_bneck=0,
        num_workers=4,
        ckpt_iter="best",
    )
    return opt


def _load_opt_oasis_ade() -> SimpleNamespace:
    opt = SimpleNamespace(
        EMA_decay=0.9999,
        add_vgg_loss=False,
        batch_size=1,
        beta1=0.0,
        beta2=0.999,
        channels_D=64,
        channels_G=64,
        checkpoints_dir="/fs/scratch/rng_cr_bcai_dl/dave2rng/oasis_ckpts/",
        continue_train=True,
        dataroot="/fs/scratch/rng_cr_bcai_dl/flash/ADEChallengeData2016",
        dataset_mode="ade20k",
        freq_fid=5000,
        freq_print=1000,
        freq_save_ckpt=20000,
        freq_save_latest=10000,
        freq_save_loss=2500,
        freq_smooth_loss=250,
        gpu_ids=0,
        lambda_labelmix=10.0,
        lambda_vgg=10.0,
        loaded_latest_iter=126199,
        lr_d=0.0004,
        lr_g=0.0001,
        name="oasis_ade20k_pretrained",
        no_3dnoise=False,
        no_EMA=False,
        no_balancing_inloss=False,
        no_flip=False,
        no_labelmix=False,
        no_spectral_norm=False,
        num_epochs=200,
        num_res_blocks=6,
        param_free_norm="syncbatch",
        phase="test",
        eed=42,
        spade_ks=3,
        which_iter="best",
        z_dim=64,
        crop_size=256,
        aspect_ratio=1.0,
        img_channels=3,
        semantic_nc=151,
        lambda_bneck=0,
        num_workers=4,
        ckpt_iter="best",
    )
    return opt


def _load_oasis(dataset: str) -> torch.nn.Module:
    if "ade" in dataset.lower():
        opt = _load_opt_oasis_ade()
    elif dataset.lower() == "cityscapes":
        opt = _load_opt_oasis_cityscapes()
    elif dataset.lower() == "coco":
        opt = _load_opt_oasis_coco()
    else:
        raise NotImplementedError
    model = models.OASIS_model(opt)
    model.eval()
    model.cuda()
    return model


def get_ganmodel(gan_name: str, dataset: str) -> torch.nn.Module:
    """Loads a pretrained GAN Model given its name and the dataset it was trained on.

    Args:
        gan_name (str): the name of the GAN model to be loaded. Options: ["OASIS"]
        dataset (str): the dataset the GAN model was trained on. Options: ["Ade20k", "Cityscapes"]

    Raises:
        NotImplementedError: if the combination of GAN Model + Training dataset is not (yet) supported

    Returns:
        torch.nn.Module: the GAN Model as object
    """
    return _load_oasis(dataset)
