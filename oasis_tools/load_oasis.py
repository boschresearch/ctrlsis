from types import SimpleNamespace
from oasis.models import models


def load_model(dataset="ade20k", wrapped=True):
    """
    This function returns a pretrained OASIS model for the specified dataset.
    wrapped = True will load OASIS with a wrapped generator.
    The wrapped generator can return any intermediate outputs.
    """

    opt = SimpleNamespace(
        EMA_decay=0.9999,
        add_vgg_loss=False,
        batch_size=4,
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
        gpu_ids="0",
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
        phase="train",
        seed=42,
        spade_ks=3,
        which_iter="best",
        z_dim=64,
        crop_size=256,
        aspect_ratio=1.0,
        img_channels=3,
        semantic_nc=151,
        lambda_bneck=0,
        num_workers=2,
    )

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
        "ade20k": {
            "semantic_nc": 151,
            "aspect_ratio": 1.0,
            "name": "oasis_ade20k_pretrained",
            "dataset_mode": "ade20k",
        },
        "celeba": {
            "semantic_nc": 19,
            "aspect_ratio": 1.0,
            "name": "20220221_celeba_main",
            "dataset_mode": "celeba",
            "checkpoints_dir": "/fs/scratch/rng_cr_bcai_dl/dave2rng/esc2rng_exp/",
            "dataroot": "/fs/scratch/rng_cr_bcai_dl_students/OpenData/CelebA-HQ-train_val/CelebA-HQ"
        },
        "cityscapes": {
            "semantic_nc": 35,
            "aspect_ratio": 2.0,
            "name": "oasis_cityscapes_pretrained",
            "dataset_mode": "cityscapes",
            "checkpoints_dir": "/fs/scratch/rng_cr_bcai_dl/dave2rng/esc2rng_exp/",
            "dataroot": "/fs/scratch/rng_cr_bcai_dl/OpenData/cityscapes",
            "EMA_decay": 0.999,
            "add_vgg_loss": False,
            "batch_size": 20,
            "beta1": 0.0,
            "beta2": 0.999,
            "channels_D": 64,
            "channels_G": 64,
            "continue_train": False,
            "freq_fid": 2500,
            "freq_print": 1000,
            "freq_save_ckpt": 20000,
            "freq_save_latest": 10000,
            "freq_save_loss": 2500,
            "freq_smooth_loss": 250,
            "gpu_ids": 0,
            "lambda_labelmix": 5.0,
            "lambda_vgg": 10.0,
            "loaded_latest_iter": 0,
            "lr_d": 0.0004,
            "lr_g": 0.0004,
            "no_3dnoise": False,
            "no_EMA": False,
            "no_balancing_inloss": False,
            "no_flip": False,
            "no_labelmix": False,
            "no_spectral_norm": False,
            "num_epochs": 200,
            "num_res_blocks": 6,
            "param_free_norm": "syncbatch",
            "phase": "test",
            "seed": 42,
            "spade_ks": 3,
            "which_iter": "best",
            "ckpt_iter": "best",
            "z_dim": 64,
            "load_size": 512,
            "crop_size": 512,
            "label_nc": 34,
            "contain_dontcare_label": True,
            "cache_filelist_read": False,
            "cache_filelist_write": False,
        },
    }

    opt.__dict__.update(specific_parameters[dataset])

    if dataset == "coco":
        if opt.phase == "test":
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

    model = models.OASIS_model(opt, wrapped=wrapped)
    
    model.eval()

    return model, opt
