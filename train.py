import subprocess
import os
import json
import copy
import torch
from torch.nn import functional as F

from oasis_tools.utils import (
    set_seed,
    label_to_one_hot,
    requires_grad,
    recursive_check,
)
from oasis_tools.visualizer import save_visualizations
from oasis.dataloaders.dataloaders import get_dataloaders
from oasis_tools.load_oasis import load_model
from tensorboardX import SummaryWriter
from visual import _generate_visual_results
from loss import (
    compute_combined_loss,
    compute_weighted_area_loss,
    compute_weighted_triplet_loss,
    compute_max_change_loss,
)
from directionmodels import CombiModel
from get_config import get_args
from utilfunctions import (
    accumulate_standing_stats,
    get_gpu_memory_map,
)

def train_step(
    args,
    iter,
    z,
    z2,
    label,
    model,
    model_optim,
    writer,
    total_iter,
    optim_classes,
    discrminator=None,
    oasis_loss=None,
):  

    """Performs one training step with loss backward call returning the GAN forward results

    Args:
        args (dict): argparse arguments. See config.py.
        iter (int): dataloader iteration step
        z (torch.Tensor): z-input noise for GAN-Model
        label (torch.Tensor): semantic label map
        model (torch.nn.Module): OASIS Model
        model_optim (torch.optim.Optimizer): model optimizer
        writer (tensorboardX.SummaryWriter): tensorboard writer
        total_iter (int): total number of iteration in the training procedure

    Returns:
        torch.Tensor: image output of GAN foward pass with z and label
    """
    if (
        args.unit_norm
        and (
            args.model_type == "param_ck"
            or args.model_type == "Az_plus_v"
            or args.model_type == "zzAzv"
        )
        and args.norm_type == "direct"
    ):  
        num_layers = 7 if args.per_layer else 1
        data = model.learned_directions.data
        data = data.view(
            1 if args.not_class_specific else args.num_classes,
            args.k,
            args.z_dim,
            num_layers,
        )
        len_dir = torch.linalg.norm(data, ord=2, dim=2, keepdim=True)
        data = data / len_dir * args.avg_l2_norm
        data = data.view(
            1 if args.not_class_specific else args.num_classes,
            args.k,
            args.z_dim * num_layers,
        )
        model.learned_directions.data = data
    model.zero_grad()
    if args.dataset == "cityscapes" or args.d_loss:
        split_bs = 1
    else:
        split_bs = 2
    z_list_1 = torch.split(z, split_bs)
    z_list_2 = torch.split(z2, split_bs)
    label_list = torch.split(label, split_bs)
    num_grad_acc_batches = len(z_list_1)
    accumulated_output_dict = {}
    keys_to_concatenate = [
        "image",
        "image_shifted",
        "loss_mask",
        "chosen_label",
        "chosen_label_img",
    ]
    keys_to_average = [
        "max_change_loss",
        "triplet_loss",
        "change_inside_mask_mean",
        "change_outside_mask_mean",
        "change_inside_mask_var",
        "change_outside_mask_var",
        "inside/outside mean ratio",
        "inside/outside var ratio",
        "area_loss",
        "combi_loss",
        "trip_pos",
        "trip_neg",
        "trip_area",
    ]
    def generate_local_noise(labelmap, z1, z2, class_id = None):
        """Generates a 3D noise tensor consisting of 2 different 1D 
        noise vectors: one for a selected class region and another 
        for every other spatial location. The function returns 
        the concatenation of the labelmap and the 3d noise tensor.

        Args:
            labelmap (torch.tensor): size(batch_size, num_classes, height, width)
            z1 (torch.tensor): size(batch_size, z_dim)
            z2 (torch.tensor): size(batch_size, z_dim)
            class_id (torch.tensor): size(batch_size, 1)
        Returns:
            torch.tensor: size(batch_size, num_classes + z_dim, height, width)
        """ 
        if not class_id:
            available_classes = labelmap.sum(dim=(2,3))
            class_id = torch.multinomial(available_classes, num_samples=1)
            class_id = class_id.view(class_id.size(0), class_id.size(1),1,1)
        index_map = labelmap.argmax(1, keepdim = True)
        mask = (index_map==class_id).float()
        noise_3d = (1-mask)*z1.view(z1.size(0),z1.size(1),1,1) + mask*z2.view(z2.size(0),z2.size(1),1,1)
        seg_plus_3D_noise = torch.cat((noise_3d, labelmap), dim = 1)
        return seg_plus_3D_noise

    for i, z1, z2, label in zip(
        range(num_grad_acc_batches), z_list_1, z_list_2, label_list
    ):
        generate_local_noise(label, z1, z2)
        out = model(
            z1,
            z2,
            label,
            allow_scaling=True,
            symmetric_directions=args.sym_loss,
            optim_classes=optim_classes,
            stochastic_triplet=args.stochastic_triplet,
            mirror_triplet=args.mirror_triplet,
            multi_triplet=args.multi_triplet,
        )
        
        if args.combined_loss_lambda != 0:
            out, combi_loss = compute_combined_loss(
                args,
                label,
                out,
                feature_difference_fct=args.diff_fct,
                weighting_type=args.weight_type,
                feature_difference_fct_inside=args.diff_fct_in,
                symmetric_directions=args.sym_loss,
                multi_triplet=args.multi_triplet,
            )
            combi_loss = combi_loss / float(num_grad_acc_batches)
            out["combi_loss"] = combi_loss.detach()
            combi_loss.backward(
                retain_graph=False,
            )

        if args.area_loss_lambda != 0:
            out, area_loss = compute_weighted_area_loss(
                args,
                label,
                out,
                feature_difference_fct=args.diff_fct,
                weighting_type=args.weight_type,
                loss_format=args.loss_format,
                scores_to_optimize=args.scores,
                feature_difference_fct_inside=args.diff_fct_in,
                symmetric_directions=args.sym_loss and not args.only_triplet_sym,
                multi_triplet=args.multi_triplet,
            )
            area_loss = args.area_loss_lambda * area_loss / float(num_grad_acc_batches)
            out["area_loss"] = area_loss.detach()
            area_loss.backward(
                retain_graph=args.triplet_loss_lambda != 0
                or args.max_change_loss_lambda != 0
                or args.d_loss
            )

        if args.triplet_loss_lambda != 0:
            out, triplet_loss = compute_weighted_triplet_loss(
                args,
                label,
                out,
                feature_difference_fct=args.diff_fct,
                weighting_type=args.weight_type,
                feature_difference_fct_inside=args.diff_fct_in,
                symmetric_directions=args.sym_loss,
                multi_triplet=args.multi_triplet,
            )
            triplet_loss = (
                args.triplet_loss_lambda * triplet_loss / float(num_grad_acc_batches)
            )
            out["triplet_loss"] = triplet_loss.detach()
            triplet_loss.backward(
                retain_graph=args.max_change_loss_lambda != 0 or args.d_loss
            )

        if args.d_loss:
            x = out["image_shifted"]
            pred = discrminator(x)
            discriminator_loss = oasis_loss.loss(pred, label, for_real=True)
            discriminator_loss = discriminator_loss * args.d_lambda
            discriminator_loss.backward()
            
        if args.max_change_loss_lambda != 0:
            out, max_change_loss = compute_max_change_loss(out)
            max_change_loss = (
                args.max_change_loss_lambda
                * max_change_loss
                / float(num_grad_acc_batches)
            )
            max_change_loss.backward()

        for key in keys_to_average:
            if key in out:
                if i == 0:
                    accumulated_output_dict[key] = 1 / num_grad_acc_batches * out[key]
                else:
                    accumulated_output_dict[key] += 1 / num_grad_acc_batches * out[key]
        for key in keys_to_concatenate:
            if key in out:
                if out[key].ndim == 4:
                    
                    out[key] = F.interpolate(
                        out[key].float(), size=(128, 128 * args.aspect_ratio)
                    ).detach()

                if i == 0:
                    accumulated_output_dict[key] = out[key]
                else:
                    accumulated_output_dict[key] = torch.cat(
                        (accumulated_output_dict[key], out[key]), dim=0
                    )
    if (
        args.model_type == "Az_plus_v" or args.model_type == "zzAzv"
    ) and args.ortho_loss:
        matrix_reg = model.learned_matrix.view(
            model.num_classes * model.k,
            model.learned_matrix.size(2),
            model.learned_matrix.size(3),
        )
        identity = torch.eye(
            model.learned_matrix.size(3), device=model.learned_matrix.device
        ).unsqueeze(0)
        ortho_loss = (
            (torch.bmm(matrix_reg, torch.transpose(matrix_reg, 1, 2)) - identity)
            .pow(2.0)
            .sum()
        )
        out["ortho_loss"] = ortho_loss.detach()
        ortho_loss.backward()
    model_optim.step()
    return accumulated_output_dict, model, model_optim, writer

def epoch_step(
    args,
    epoch,
    dataloader_train,
    model,
    model_optim,
    writer,
    total_iter,
    discrminator=None,
    oasis_loss=None,
    dataloader_test=None,
):  
    """Performs one epoch training step saving the current model after each epoch iteration.
    It also saves visualizations and log metrics.

    Args:
        args (dict): argparse arguments. See config.py.
        epoch (int): epoch counter
        dataloader_train (torch.utils.data.DataLoader): trainig dataloader
        model (torch.nn.Module): OASIS Model
        model_optim (torch.optim.Optimizer): model optimizer
        writer (tensorboardX.SummaryWriter): tensorboard writer
        total_iter (int): total number of iteration in the training procedure
    """
    optim_classes = torch.tensor(args.optim_classes, device=args.device)
    num_steps_per_epoch = len(dataloader_train)
    for iter, batch in enumerate(dataloader_train):
        if total_iter % 100 == 0:
            if args.asymmetric_training:
                ckpt_name = f"ep{epoch}_it{total_iter}.pth"
                state_dict_path = os.path.join(args.latest_ckpt_folder, ckpt_name)
                torch.save(
                    model.state_dict(),
                    os.path.join(state_dict_path),
                )
                print(f"Saved {state_dict_path}")
                prev_ckpts = os.listdir(args.latest_ckpt_folder)
                prev_ckpts = [p for p in prev_ckpts if p != ckpt_name]
                for ckpt in prev_ckpts:
                    prev_ckpt_path = os.path.join(args.latest_ckpt_folder, ckpt)
                    subprocess.run(f"rm {prev_ckpt_path}", shell=True)
                    print(f"Removed {prev_ckpt_path}")
        total_iter += 1
        image, label = batch["image"].to(args.device), batch["label"].to(args.device)
        label = label_to_one_hot(
            label.long(), num_classes=args.num_classes
        )  
        z = torch.randn(args.batch_size, args.z_dim, device=args.device)
        z2 = torch.randn(args.batch_size, args.z_dim, device=args.device)
        outputs, model, model_optim, writer = train_step(
            args,
            iter,
            z,
            z2,
            label,
            model,
            model_optim,
            writer,
            total_iter,
            optim_classes=optim_classes,
            discrminator=discrminator,
            oasis_loss=oasis_loss,
        )

        if (iter + 1) % args.print_every == 0 or args.test:
            if "change_inside_mask_mean" in outputs:
                inside = "inside " + str(outputs["change_inside_mask_mean"]) + ", "
            else:
                inside = ""
            if "change_outside_mask_mean" in outputs:
                outside = "outside " + str(outputs["change_outside_mask_mean"]) + ", "
            else:
                outside = ""
            if "area_loss" in outputs:
                area_loss = "area_loss " + str(outputs["area_loss"]) + ", "
            else:
                area_loss = ""
            if "triplet_loss" in outputs:
                triplet_loss = "triplet_loss " + str(outputs["triplet_loss"]) + ", "
            else:
                triplet_loss = ""
            if "combi_loss" in outputs:
                combi_loss = "combi_loss " + str(outputs["combi_loss"]) + ", "
            else:
                combi_loss = ""  
            if "max_change_loss" in outputs:
                max_change_loss = (
                    "max_change_loss " + str(outputs["max_change_loss"]) + ", "
                )
            else:
                max_change_loss = ""

            if "trip_pos" in outputs:
                trip_pos = "trip_pos " + str(outputs["trip_pos"]) + ","
            else:
                trip_pos = ""

            if "trip_neg" in outputs:
                trip_neg = "trip_neg " + str(outputs["trip_neg"]) + ","
            else:
                trip_neg = ""

            if "trip_area" in outputs:
                trip_area = "trip_area " + str(outputs["trip_area"]) + ","
            else:
                trip_area = ""
            print_str = (
                str(get_gpu_memory_map())
                + f" ep {epoch} it {iter} / {num_steps_per_epoch}: {inside}{outside}{area_loss}{triplet_loss}{max_change_loss}{combi_loss}{trip_pos}{trip_neg}{trip_area}"
            )
            print(print_str, flush=True)

        if (iter + 1) % args.write_every == 0 or args.test:
            scalars_to_track = [
                "area_loss",
                "triplet_loss",
                "max_change_loss",
                "change_inside_mask_mean",
                "change_outside_mask_mean",
                "inside/outside mean ratio",
                "change_inside_mask_var",
                "change_outside_mask_var",
                "inside/outside var ratio",
                "weights_mean",
                "weights_var",
                "combi_loss",
                "trip_pos",
                "trip_neg",
                "trip_area",
            ]
            for scalar in scalars_to_track:
                if scalar in outputs:
                    writer.add_scalar(scalar, outputs[scalar], total_iter)

        if (iter + 1) % args.big_vis_every == 0 or args.test:
            vis_dir = os.path.join(args.output_dir, f"ep_{epoch}_it_{iter}")
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)

            if args.dataset == "ade20k":
                image_list = [
                    "ADE_val_00001953",  
                    "ADE_train_00017081",  
                    "ADE_val_00000797",  
                    "ADE_train_00011962",  
                ]
                alpha_image_list = [
                    "ADE_train_00017081",  
                    "ADE_val_00000797",  
                    "ADE_val_00001953",  
                ]
                args.num_z = 1
                args.load_images_directly = True
                args.global_shift = False
                args.num_images = 5
                args.cutmix = False
                allowed_classes = args.optim_classes
            elif args.dataset == "cityscapes":
                args.num_z = 1
                allowed_classes = [6, 7, 11, 21, 26, 28]
                args.load_images_directly = False
                image_list = None
                alpha_image_list = None
                args.global_shift = False
                args.max_num_classes = 20
                args.num_images = 1
                args.cutmix = False
            elif args.dataset == "celeba":
                args.num_z = 3
                allowed_classes = args.optim_classes
                args.load_images_directly = False
                image_list = None
                alpha_image_list = None
                args.global_shift = False
                args.max_num_classes = 20
                args.num_images = 1
                args.cutmix = False
            else:
                allowed_classes = args.optim_classes

            _generate_visual_results(
                main_args=args,
                image_list=image_list,
                alpha_image_list=alpha_image_list,
                direction_model=model,
                dataloader=copy.deepcopy(dataloader_test),
                allowed_classes=allowed_classes,
                list_of_k=[k_idx for k_idx in range(args.k)],
                exp_save_folder=vis_dir,
                num_classes=args.num_classes,
                z_dim=64,
                device=args.device,
                image_generator=model.generator,
                cut_mix=False,
            )

        if (iter + 1) % args.vis_every == 0 or args.test:
            visuals_dict = {
                "labels": (
                    F.interpolate(label.float(), size=(128, 128 * args.aspect_ratio)),
                    "label",
                ),
                "real": (
                    F.interpolate(image.float(), size=(128, 128 * args.aspect_ratio)),
                    "rgb",
                ),
                "fake": (outputs["image"], "rgb"),
                "fake shift": (outputs["image_shifted"], "rgb"),
            }
            if "loss_mask" in outputs:
                visuals_dict["loss_mask"] = (
                    F.interpolate(
                        outputs["loss_mask"]
                        * outputs["chosen_label"].view(-1, 1, 1, 1),
                        size=(128, 128 * args.aspect_ratio),
                    ),
                    "label",
                )
            if "chosen_label_img" in outputs:
                visuals_dict["chosen_label"] = (outputs["chosen_label_img"], "label")
            for key in visuals_dict.keys():
                visuals_dict[key] = list(visuals_dict[key])
                visuals_dict[key][0] = visuals_dict[key][0][:6]
            save_visualizations(
                visuals_dict,
                save_folder=args.output_dir,
                epoch=epoch,
                iter=iter,
                name="batch",
            )
        if args.test:
            break
    return model, model_optim, writer, total_iter

def train(args):
    """Triggers main training procedure for discovering OASIS semantic directions
    Args:
        args (dict): argparse arguments. See config.py.
    """
    set_seed(args.seed)
    print(args.dataset)
    oasis, oasis_opt = load_model(dataset=args.dataset, wrapped=True)
    if args.optim_classes is not None and args.dataset == "ade20k":
        with open(
            "assets/which_images_contain_which_label_train_set.json", "r"
        ) as read_file:
            image_lookup_train = json.load(read_file)
        allowed_images_train = set()
        for lbl in args.optim_classes:
            allowed_images_train = allowed_images_train.union(
                image_lookup_train[str(lbl)]
            )
            print(lbl, len(image_lookup_train[str(lbl)]), len(allowed_images_train))
        oasis_opt.image_list_train = list(allowed_images_train)
        for i in range(len(oasis_opt.image_list_train)):
            oasis_opt.image_list_train[i] = (
                oasis_opt.image_list_train[i].replace(".jpg", "").replace(".png", "")
            )
        with open(
            "assets/which_images_contain_which_label_test_set.json", "r"
        ) as read_file:
            image_lookup_val = json.load(read_file)
        allowed_images_val = set()
        for lbl in args.optim_classes:
            allowed_images_val = allowed_images_val.union(image_lookup_val[str(lbl)])
            print(lbl, len(image_lookup_val[str(lbl)]), len(allowed_images_val))
        oasis_opt.image_list_val = list(allowed_images_val)
        for i in range(len(oasis_opt.image_list_val)):
            oasis_opt.image_list_val[i] = (
                oasis_opt.image_list_val[i].replace(".jpg", "").replace(".png", "")
            )
    oasis = oasis.to(args.device)
    oasis_opt.batch_size = args.batch_size
    oasis_opt.phase = "train"
    oasis_opt.continue_train = True
    oasis_opt.num_workers = 2
    dataloader_train, _ = get_dataloaders(oasis_opt)
    oasis_opt_acc = copy.deepcopy(oasis_opt)
    oasis_opt_acc.batch_size = 2
    dataloader_acc, _ = get_dataloaders(oasis_opt_acc)

    if (
        args.dataset == "cityscapes"
        or args.dataset == "coco"
        or args.dataset == "ade20k"
        or args.dataset == "celeba"
    ):
        oasis_opt_test = copy.deepcopy(oasis_opt)
        oasis_opt_test.phase = "test"
        oasis_opt_test.batch_size = 1
        print(oasis_opt_test)
        dataloader_test, _ = get_dataloaders(oasis_opt_test)
    else:
        dataloader_test = None

    discrminator = None
    oasis_loss = None
    print("Loading OASIS")
    args.num_layers = 7
    generator = oasis.netEMA.to(args.device)
    generator = accumulate_standing_stats(
        args,
        generator,
        dataloader_acc,
        max_iter=10 if args.test else args.acc_iter,
    )
    generator = generator.eval()
    requires_grad(generator, False)
    model = CombiModel(
        generator=generator,
        batch_size=args.batch_size,
        z_dim=64,
        num_classes=args.num_classes,
        k=args.k,
        c=args.c,
        num_layers=args.num_layers,
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
        alpha_flipping=args.alpha_flipping,
        merge_classes=args.merge_classes and args.dataset == "celeba",
        rotation_alpha=args.rotation_alpha,
        translation_alpha=args.translation_alpha,
        zzAzv_alpha=args.zzAzv_alpha,
        flip_zzAzv=args.flip_zzAzv,
    )

    model = model.to(args.device)
    start_epoch = 0
    total_iter = 0
    if args.asymmetric_training:
        if not os.path.exists(args.latest_ckpt_folder):
            os.makedirs(args.latest_ckpt_folder)
        ckpts = os.listdir(args.latest_ckpt_folder)
        if len(ckpts) > 0:
            latest_ckpt = sorted(ckpts)[-1]
            state_dict_path = os.path.join(args.latest_ckpt_folder, latest_ckpt)
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            print("Loaded state dict", state_dict_path)
            start_epoch = int(latest_ckpt.split("ep")[1].split("_")[0])
            total_iter = int(latest_ckpt.split("it")[1].split(".")[0])

    elif args.ckpt_path != "":
        state_dict = torch.load(args.ckpt_path)
        model.load_state_dict(state_dict)
        print("Loaded state dict", args.ckpt_path)

    print("model.parameters()")
    for m in model.parameters():
        if m.requires_grad:
            print(m.size())

    if args.optim_type == "adamw":
        model_optim = torch.optim.AdamW(
            [m for m in model.parameters() if m.requires_grad],
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
        )
    elif args.optim_type == "adam":
        model_optim = torch.optim.Adam(
            [m for m in model.parameters() if m.requires_grad],
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
        )
    elif args.optim_type == "sgd":
        model_optim = torch.optim.SGD(
            [m for m in model.parameters() if m.requires_grad], lr=args.lr
        )
    else:
        raise NotImplementedError

    writer = SummaryWriter(args.output_dir)

    for epoch in range(start_epoch, args.num_epochs):
        if args.asymmetric_training:
            ckpt_name = f"ep{epoch}_it{total_iter}.pth"
            state_dict_path = os.path.join(args.latest_ckpt_folder, ckpt_name)
            torch.save(
                model.state_dict(),
                os.path.join(state_dict_path),
            )
            print(f"Saved {state_dict_path}")
            prev_ckpts = os.listdir(args.latest_ckpt_folder)
            prev_ckpts = [p for p in prev_ckpts if p != ckpt_name]
            for ckpt in prev_ckpts:
                prev_ckpt_path = os.path.join(args.latest_ckpt_folder, ckpt)
                subprocess.run(f"rm {prev_ckpt_path}", shell=True)
                print(f"Removed {prev_ckpt_path}")

        model, model_optim, writer, total_iter = epoch_step(
            args,
            epoch,
            dataloader_train,
            model,
            model_optim,
            writer,
            total_iter,
            discrminator=discrminator,
            oasis_loss=oasis_loss,
            dataloader_test=dataloader_test,
        )

        if epoch % args.ckpt_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"combi_model_ep_{epoch}.pth"),
            )
            subprocess.run(f"chmod -R 777 {args.output_dir}", shell=True)

            if args.dataset == "ade20k":
                image_list = [
                    "ADE_val_00001953",  
                    "ADE_train_00017081",  
                    "ADE_val_00000797",  
                    "ADE_train_00011962",  
                ]
                alpha_image_list = []
                args.num_z = 1
                args.load_images_directly = True
                args.global_shift = False
                args.num_images = 5
                args.cutmix = False
                allowed_classes = args.optim_classes
            elif args.dataset == "cityscapes":
                args.num_z = 1
                allowed_classes = [6, 7, 11, 21, 26, 28]
                args.load_images_directly = False
                image_list = None
                alpha_image_list = None
                args.global_shift = False
                args.max_num_classes = 20
                args.num_images = 1
                args.cutmix = False
            elif args.dataset == "celeba":
                args.num_z = 3
                allowed_classes = args.optim_classes
                args.load_images_directly = False
                image_list = None
                alpha_image_list = None
                args.global_shift = False
                args.max_num_classes = 20
                args.num_images = 1
                args.cutmix = False
            else:
                allowed_classes = args.optim_classes
            vis_dir = os.path.join(args.output_dir, f"ep_{epoch}")
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            _generate_visual_results(
                main_args=args,
                image_list=image_list,
                alpha_image_list=alpha_image_list,
                direction_model=model,
                dataloader=copy.deepcopy(dataloader_test),
                allowed_classes=allowed_classes,
                list_of_k=[k_idx for k_idx in range(args.k)],
                exp_save_folder=vis_dir,
                num_classes=args.num_classes,
                z_dim=64,
                device=args.device,
                image_generator=model.generator,
                cut_mix=False,
            )

        if args.test:
            break

if __name__ == "__main__":
    args = get_args()
    train(args)
