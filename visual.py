"""
This script evaluates the learned latent directions.
It computes the consistency and diversity score.
Specify a yaml in ./config_files. 
for example: 1801_ablation.yaml 

python visual.py \
    --load_images_directly \
    --config 0000.yaml \
    --phase test \
    --algorithm ours \
    --name vis \
    --dataset ade20k \
    --topc all \
    --img_generator_name oasis \
    --max_k 5 \
    --num_z 1 \
    --acc_iter 10 \
    --topc all \
    --max_num_classes 10

python visual.py \
    --num_images 2000 \
    --config 0000.yaml \
    --phase test \
    --algorithm ours \
    --name vis \
    --dataset ade20k \
    --topc all \
    --img_generator_name oasis \
    --max_k 5 \
    --num_z 1 \
    --acc_iter 500 \
    --topc all \
    --max_num_classes 10 \
    --load_images_directly --global_shift 
"""

import argparse
import os

import torch
from torch import nn
from oasis.dataloaders.dataloaders import get_dataloaders
from oasis_tools.utils import set_seed, label_to_one_hot, requires_grad
from oasis_tools.load_oasis import load_model
from directionmodels import CombiModel
import json
from types import SimpleNamespace
from utilfunctions import (
    get_top_c_classes,
    min_max_scaling,
    apply_shift,
    random_string,
    write_on_image,
    accumulate_standing_stats,
)
from torch.nn import functional as F
from torchvision.utils import save_image
from argparse import ArgumentParser
import yaml
from oasis_tools.utils import recursive_check
import copy
import warnings
from oasis_tools.visualizer import colorize_label

avg_l2_norm = torch.linalg.norm(torch.randn(10000, 64), ord=2, dim=1).mean().item()

def _generate_visual_results(
    main_args: argparse.Namespace,
    image_list: list,
    alpha_image_list: list,
    direction_model: CombiModel,
    dataloader: torch.utils.data.dataloader.DataLoader,
    allowed_classes: list,
    list_of_k: list,
    exp_save_folder: str,
    num_classes=151,
    z_dim=64,
    device="cuda:0",
    image_generator: torch.nn.Module = None,
    cut_mix=False,
    max_classes=None,
    folder="",
    experiment_key="",
    cumulative_image_folder="/fs/scratch/rng_cr_bcai_dl/jea2rng/teaser9_cuts/",
    negative=False,
    save_cumulative=False,
) -> None:
    """[Generates images and saves them to disc]

    Args:
        main_args (argparse.Namespace): parsed arguments
        image_list (list): images to visualize different k for different classes
        alpha_image_list (list): images for visualizing different alphas
        direction_model (CombiModel): trained model for latent directions
        dataloader (torch.utils.data.dataloader.DataLoader): dataloader to provide images
        allowed_classes (list): for which classes to visualize results
        list_of_k (list): for which k to visualize results
        exp_save_folder (str): where to put the resulting images
        num_classes (int, optional): number of classes specific to used dataset. Defaults to 151.
        z_dim (int, optional): latent dimensionality. Defaults to 64.
        device (str, optional): which gpu to use. Defaults to "cuda:0".
        image_generator (nn.Module, optional): generator to convert noise to images. Defaults to None.
        cut_mix (bool, optional): cut and mix results instead of cutting and mixing noise. Defaults to False.
    """

    set_seed(0)
    if "alpha_for_vis" in main_args.__dict__:
        alpha_for_vis = main_args.alpha_for_vis
    else:
        alpha_for_vis = (
            torch.linalg.norm(torch.randn(10000, main_args.z_dim), ord=2, dim=1)
            .mean()
            .item()
        )
    if negative:
        alpha_for_vis = -copy.deepcopy(alpha_for_vis)

    if main_args.model_type == "Az_plus_v" or main_args.model_type == "zzAzv":

        if "rotation_alpha" in main_args.__dict__:
            rotation_alpha = main_args.rotation_alpha
        else:
            rotation_alpha = None

        if "translation_alpha" in main_args.__dict__:
            translation_alpha = main_args.translation_alpha
        else:
            translation_alpha = None

        if "zzAzv_alpha" in main_args.__dict__:
            zzAzv_alpha = main_args.zzAzv_alpha
        else:
            zzAzv_alpha = None
    else:
        rotation_alpha = 1
        translation_alpha = 1
        zzAzv_alpha = alpha_for_vis

    notifications_printed = False
    z_list = [torch.randn(1, z_dim, device=device) for _ in range(main_args.num_z)]
    torch.save(z_list, os.path.join(exp_save_folder, "noise.pth"))

    if main_args.dataset == "cityscapes":  
        main_args.load_images_directly = False
    if main_args.load_images_directly:
        images = image_list
        tmp_dataloader = (
            dataloader.dataset.get_single_item(img_name) for img_name in images
        )
    else:
        tmp_dataloader = dataloader
 
    alpha_counter = 0
    if save_cumulative:
        root = os.path.join(
            cumulative_image_folder, main_args.dataset, main_args.img_generator_name
        )
    model_id = main_args.direction_model + "_" + folder.split("/")[-1].split("-")[0]
    if negative:
        model_id += "_negative"

    print("alpha_for_vis", alpha_for_vis)

    for i, batch in enumerate(tmp_dataloader):
        print(f"> {i} {batch['label'].size()}", flush=True)
        img_name = batch["name"]
        if isinstance(img_name, list):
            assert len(img_name) == 1
            img_name = img_name[0]

        if save_cumulative:
            img_folder = os.path.join(
                root, img_name.replace(".png", "").replace(".jpg", "")
            )

            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        if main_args.dataset == "cityscapes" and not main_args.load_images_directly:
            img_name = batch["name"]
            for j in range(len(img_name)):
                img_name[j] = img_name[j].replace(".png", "")

        if main_args.load_images_directly:
            visualize_alpha = img_name in alpha_image_list
        else:
            visualize_alpha = alpha_counter < 0  

        if visualize_alpha:
            alpha_counter += 1
        if isinstance(img_name, list):
            assert len(img_name) == 1, "Code requires to see only one image at a time."
            img_name = img_name[0]
        if ".jpg" not in img_name:
            img_name += ".jpg"

        _, label = batch["image"].to(device), batch["label"].to(device)

        if label.ndim == 3:
            label = label.unsqueeze(1)
        label = label_to_one_hot(
            label.long(), num_classes=num_classes
        )  

        unique = label.argmax(1).unique()

        if 0 in unique and len(unique) > 1:
            unique = unique[1:]
        unique_set = set(unique.cpu().numpy())
        classes = unique_set.intersection(set(allowed_classes))
        classes = sorted(classes)
 
        rgbs = []
        class_counter = 0
        if len(classes) > 0:
            for c in classes:
                if class_counter >= main_args.max_num_classes:
                    break
                class_counter += 1

                labelmap = colorize_label(label.argmax(1), 200)
                if save_cumulative and c == classes[0]:
                    lbl_path = os.path.join(img_folder, "lbl.png")
                    save_image(labelmap, lbl_path)

                alpha_img_name = f"alpha_class_{c}_{img_name}"

                for z_i in range(main_args.num_z):
                    rgbs.append(labelmap.to("cpu"))

                    num_columns = 1

                    label_tensor = (
                        torch.ones((label.size(0),), device=label.device) * c
                    ).long()

                    if visualize_alpha and z_i == 0:
                        alpha_image_tensors = []
                        alpha_list = [-24, -16, -8, 4, 0, 4, 8, 16, 24]

                    for k_idx, k in enumerate(list_of_k):
                        if main_args.global_shift:
                            assert not (
                                main_args.model_type == "Az_plus_v"
                                or main_args.model_type == "zzAzv"
                            )
                            direction = direction_model.get_random_directions_3d(
                                label_map=label,
                                alpha=1.0,
                                z=z_list[z_i],
                                use_all_labels=True,
                                rand_k_per_class=True,
                            ).to(label.device)
                            label_tensor = None
                        else:
                            direction = direction_model.get_direction(
                                k=k, c=label_tensor, alpha=1.0, z=z_list[z_i]
                            )  
                            if (
                                main_args.model_type == "Az_plus_v"
                                or main_args.model_type == "zzAzv"
                            ):
                                matrix = direction_model.learned_matrix[
                                    c, k
                                ]  
                                
                            else:
                                matrix = None
                        if visualize_alpha and z_i == 0:
                            with torch.no_grad():
                                for alpha_iter, alpha in enumerate(alpha_list):
                                    alpha_out = apply_shift(
                                        z=z_list[z_i],
                                        direction=direction,
                                        label=label,
                                        image_generator=image_generator,
                                        chosen_label=label_tensor,
                                        num_layers=main_args.num_layers,
                                        use_class_mask=True,
                                        alpha=translation_alpha
                                        if zzAzv_alpha is not None
                                        else alpha,
                                        matrix=matrix,
                                        rotation_alpha=rotation_alpha,
                                        zzAzv_alpha=alpha,
                                    )
                                    if alpha_iter == 0:
                                        mask_img = colorize_label(
                                            alpha_out["noise_mask"], 2
                                        )
                                        alpha_image_tensors.append(
                                            write_on_image(
                                                f"k={k}",
                                                mask_img,
                                                font_pixel_height=40,
                                                color="white",
                                            )
                                        )  
                                    img = min_max_scaling(alpha_out["image_shifted"])
                                    img = write_on_image(
                                        f"{alpha}",
                                        img,
                                        font_pixel_height=40,
                                        color="black",
                                        transparent=False,
                                    )
                                    alpha_image_tensors.append(img)  
                            if main_args.dataset == "cityscapes":
                                for r in range(len(alpha_image_tensors)):
                                    if alpha_image_tensors[r].size(2) == 128:
                                        alpha_image_tensors[r] = F.interpolate(
                                            alpha_image_tensors[r], size=(256, 512)
                                        )

                            alpha_img_save_tensor = torch.cat(
                                alpha_image_tensors, dim=0
                            )

                        with torch.no_grad():
                            out = apply_shift(
                                z=z_list[z_i],
                                direction=direction,
                                label=label,
                                image_generator=image_generator,
                                chosen_label=label_tensor,
                                num_layers=main_args.num_layers,
                                use_class_mask=True,
                                alpha=translation_alpha,
                                matrix=matrix,
                                rotation_alpha=rotation_alpha,
                                zzAzv_alpha=zzAzv_alpha,
                            )
                            mask = out["noise_mask"]

                        rgb = out["image"]
                        if k_idx == 0:
                            lblmap_img = colorize_label(mask, 2).to("cpu")
                            lblmap_img = write_on_image(str(c), lblmap_img)
                            rgbs.append(lblmap_img)
                            rgb = (rgb + 1) / 2
                            rgbs.append(rgb.to("cpu"))
                            num_columns += 2
                            if save_cumulative:
                                mask_path = os.path.join(img_folder, f"lbl_{c}.png")
                                save_image(lblmap_img, mask_path)
                                im_name = f"{z_i}_orig_{model_id}.png"
                                lbl_path = os.path.join(img_folder, im_name)
                                save_image(rgb, lbl_path)
                                
                        if cut_mix:
                            if not notifications_printed:
                                print("--> USING CUTMIX", flush=True)

                            rgb_shifted = out["image_shifted_no_mask"] * mask + out[
                                "image"
                            ] * (1 - mask)
                        else:
                            rgb_shifted = out["image_shifted"]

                        rgb_shifted = (rgb_shifted + 1) / 2
                        
                        rgbs.append(rgb_shifted.to("cpu"))
                        num_columns += 1

                        if save_cumulative:
                            im_name = f"{z_i}_{k}_{c}_{model_id}.png"
                            
                            lbl_path = os.path.join(img_folder, im_name)
                            
                            save_image(rgb_shifted, lbl_path)

                    if visualize_alpha and z_i == 0:
                        alpha_save_path = os.path.join(exp_save_folder, alpha_img_name)
                        print(f"alpha: saving {alpha_save_path}", flush=True)

                        if (
                            main_args.dataset == "cityscapes"
                            and not main_args.load_images_directly
                        ):
                            folder_path = "/".join(alpha_save_path.split("/")[:-1])
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                        save_image(
                            alpha_img_save_tensor,
                            alpha_save_path,
                            nrow=len(alpha_list) + 1,
                            normalize=True,
                        )

            if main_args.dataset == "cityscapes":
                for r in range(len(rgbs)):
                    if rgbs[r].size(2) == 128:
                        rgbs[r] = F.interpolate(rgbs[r], size=(256, 512))

            rgbs = torch.cat(rgbs, dim=0)
            save_name = img_name
            if negative:
                save_name = save_name.replace(".jpg", "_negative.jpg")
            file_name = os.path.join(exp_save_folder, save_name)
            if main_args.dataset == "cityscapes" and not main_args.load_images_directly:
                folder_path = "/".join(file_name.split("/")[:-1])
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

            print(f"saving {file_name}", flush=True)
            save_image(rgbs, file_name, nrow=num_columns, normalize=True)
            if not notifications_printed:
                notifications_printed = True

        if i > main_args.num_images and not main_args.load_images_directly:
            break


def _visualize_ours(
    main_args: argparse.Namespace,
    dataloader: torch.utils.data.dataloader.DataLoader,
    image_model: nn.Module,
    image_generator: nn.Module,
):
    """Visualize results for our own trained models.

    Loads a list of models from a yaml file. Iterates over this list
    to visualize different k for different classes,
    as well as different alphas.

    Args:
        main_args (argparse.Namespace): Parsed arguments.
        dataloader (torch.utils.data.dataloader.DataLoader): Dataloader.
        image_model (nn.Module): Full semantic image synthesis model (including generator, discriminator, other networks)
        image_generator (nn.Module): Pure generator object with forward pass to generate images from noise.
    """

    with open(f"./config_files/{main_args.config}", "r") as file:
        yaml_dict = yaml.safe_load(file)

        results_dir = yaml_dict["results_dir"]
        experiments = yaml_dict["experiments"]
        ckpt = yaml_dict["ckpt"]
        save_folder = yaml_dict["save_folder"]
        image_list = yaml_dict["images"]
        if "alpha_images" in yaml_dict:
            alpha_image_list = yaml_dict["alpha_images"]
        else:
            alpha_image_list = []
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    experiment_folders = os.listdir(results_dir)
    experiment_folders = [
        folder
        for folder in experiment_folders
        if any(exp in folder for exp in experiments)
    ]
    experiment_folders = [
        os.path.join(results_dir, folder) for folder in experiment_folders
    ]

    set_seed(0)

    for folder in experiment_folders:

        experiment_key = (
            f"{main_args.algorithm }_{main_args.img_generator_name}_{main_args.dataset}_{main_args.name}_"
            + folder.split("/")[-1][:32]
            + "_"
            + ckpt.replace(".pth", "").replace("combi_model_", "")
        )
        if main_args.global_shift:
            experiment_key += "_global_shift"

        state_dict_path = os.path.join(folder, ckpt)
        print("state_dict_path", state_dict_path)
        if not os.path.exists(state_dict_path):
            print(state_dict_path, "does not exist")
            continue

        with open(os.path.join(folder, "args.json"), "r") as file:
            args_dict = json.load(file)

        args = SimpleNamespace()
        args.__dict__.update(args_dict)

        if not main_args.dataset == args.dataset:
            warnings.warn("Dataset of ckpt does not match specified dataset.")
            continue

        args.batch_size = 1

        set_seed(args.seed)

        model = CombiModel(
            generator=image_generator,
            batch_size=args.batch_size,
            z_dim=64,
            num_classes=args.num_classes,
            k=args.k,
            c=args.c,
            num_layers=7,
            pred_all_layers=args.per_layer,
            model_type=args.model_type,
            use_class_mask=args.use_class_mask,
            unit_norm=args.unit_norm,
            normalize_images=args.normalize_images,
            alpha_scaling=args.alpha_scaling,
            feature_name=args.feature_name,
            avg_l2_norm=args.avg_l2_norm,
            norm_type=args.norm_type,
            not_class_specific=args.not_class_specific,
            learn_per_layer_weights=args.per_layer,
            rotation_alpha=args.rotation_alpha,
            translation_alpha=args.translation_alpha,
            zzAzv_alpha=args.zzAzv_alpha,
            flip_zzAzv=args.flip_zzAzv,
        )
        model = model.to(args.device)
        state_dict = torch.load(state_dict_path)
        
        print(" \n\n")
        print(" \n\n")

        if True:
            new_state_dict = {}

            for key in state_dict.keys():
                if not "oasis.netG" in key and not "oasis.netD" in key:
                    new_key = key.replace("oasis.netEMA", "generator")
                    
                    new_state_dict[new_key] = state_dict[key]
                if "learned_directions" in key:
                    new_state_dict[key] = state_dict[key]

            state_dict = new_state_dict
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        acc_stats = True
        if acc_stats:
            print("OURS: ACC STATS")
            model.generator = accumulate_standing_stats(
                args,
                model.generator,
                copy.deepcopy(dataloader),
                max_iter=main_args.acc_iter,
            )
            model.generator.eval()
        else:
            print("OURS: NOT ACC STATS")

        exp_save_folder = os.path.join(
            save_folder,
            experiment_key,  
        )

        if not os.path.exists(exp_save_folder):
            os.makedirs(exp_save_folder)

        if len(main_args.topc) == 0:
            topc = get_top_c_classes(args.c)
        else:
            topc = main_args.topc

        if 0 in topc:
            print(" \nWarning: topc contains UNK class. Removing it now.\n ")
            topc = [top_idx for top_idx in topc if top_idx != 0]
        eval_labels = topc
        num_eval_classes = len(eval_labels)
        print(f"num eval classess: {num_eval_classes}")
        print(f"eval classes: {eval_labels}")

        num_k = min(main_args.max_k, args.k)

        _generate_visual_results(
            main_args,
            image_list,
            alpha_image_list,
            direction_model=model,
            dataloader=dataloader,
            list_of_k=[k for k in range(num_k)],
            allowed_classes=topc,
            num_classes=args.num_classes,
            z_dim=args.z_dim,
            device=args.device,
            exp_save_folder=exp_save_folder,
            image_generator=image_generator,
            cut_mix=main_args.cutmix,
            folder=folder,
            experiment_key=experiment_key,
        )

def _main(args):
    """Load dataloader and image generator.

    Args:
        args (argpars.NameSpace): Command line arguments.
    """
    if args.dataset == "ade20k":
        args.num_classes = 151
        args.aspect_ratio = 1
        args.c = args.num_classes

    elif args.dataset == "cityscapes":
        args.num_classes = 35
        args.aspect_ratio = 2
        args.c = args.num_classes

    elif args.dataset == "coco":
        args.num_classes = 183
        args.aspect_ratio = 1
        args.c = args.num_classes

    if args.topc == "":
        args.topc = []
    elif args.topc == "all":
        args.topc = [e for e in range(args.num_classes)]
    else:
        args.topc = [int(elem) for elem in args.topc.split(",")]
    assert len(args.config) > 0, "Specify yaml file."

    args.batch_size = 1
    args.seed = 0
    args.device = "cuda:0"
    args.k = args.max_k

    set_seed(args.seed)

    oasis, oasis_opt = load_model(dataset=args.dataset, wrapped=True)
    oasis_opt.batch_size = args.batch_size
    oasis_opt.phase = args.phase
    dataloader_train, _ = get_dataloaders(oasis_opt)

    print("loading oasis")
    args.z_dim = 64
    args.num_layers = 7
    oasis = oasis.to(args.device)

    full_model = oasis
    image_generator = oasis.netEMA
    image_generator.eval()
    acc_stats = False
    if acc_stats:
        requires_grad(image_generator, True)

        image_generator = accumulate_standing_stats(
            args,
            image_generator,
            dataloader_train,
            max_iter=args.acc_iter,
        )
        image_generator.eval()
    else:
        print("NOT ACC STATS")

    requires_grad(oasis, False)
    requires_grad(image_generator, False)

    if args.algorithm == "ours":
        print("eval ours")
        _visualize_ours(args, dataloader_train, full_model, image_generator)
    else:
        print("eval related work")
        _visualize_related_work(args, dataloader_train, full_model, image_generator)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Name of config yaml in config_files folder",
    )

    parser.add_argument("--num_z", type=int, default=5, help="JHow many z to visualize")
    parser.add_argument("--phase", default="test", choices=["train", "test"])
    parser.add_argument(
        "--algorithm", default="ours", choices=["ours"]
    )
    parser.add_argument(
        "--topc", type=str, default="", help="examples: 1,2,3 or 24 or 4,8"
    )
    parser.add_argument("--dataset", type=str, default="ade20k")
    parser.add_argument("--img_generator_name", type=str, default="oasis")
    parser.add_argument(
        "--max_k",
        type=int,
        default=8,
        help="Chooses the first max_k directions of any method",
    )
    parser.add_argument("--max_num_classes", type=int, default=5)

    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--acc_iter", type=int, default=500)
    parser.add_argument("--alpha_for_vis", type=int, default=7.96)
    parser.add_argument("--load_images_directly", action="store_true", default=False)
    parser.add_argument(
        "--direction_model",
        type=str,
        default="ours",
        choices=["ours"],
    )
    parser.add_argument(
        "--name", type=str, default="", help="string is added to name of save folder"
    )
    parser.add_argument(
        "--cutmix", action="store_true", default=False, help="applying mask afterwards"
    )
    parser.add_argument(
        "--global_shift",
        action="store_true",
        default=False,
        help="applying k to all labels simultanously",
    )

    _main(parser.parse_args())
