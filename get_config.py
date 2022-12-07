import json
import os
from datetime import datetime
from argparse import ArgumentParser
from oasis_tools.layer_names import LAYER_LOOKUP 
import torch

def get_args():
    model_types = """direction models are supported with different input configurations \n
    such as c (
    - param_ck: the directions are the trainable parameters itself. There are c * k directions. \n
    - net_ck: a mlp network generates the direction in dependence of c, k. \n
    - net_ckz: a mlp network generates the direction in dependence of c, k, z.
    """

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of directions models per class to be discovered",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=None,
        help="Number of classes to be considered. \
                        The top c classes per Avg Pixel Frequency are taken",
    )
    parser.add_argument("--dataset", type=str, default="ade20k") 
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--name", type=str, default="combi", help="Experiment name for logging"
    )
    parser.add_argument("--vis_every", type=int, default=10)
    parser.add_argument("--big_vis_every", type=int, default=999000)
    parser.add_argument(
        "--print_every", type=int, default=10, help="Frequency of printing"
    )
    parser.add_argument(
        "--write_every",
        type=int,
        default=10,
        help="Frequency of writing to TF Record file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="param_ck",
        help=model_types,
        choices=[
            "param_ck",
            "net_ck",
            "net_ckz",
            "shared_param_ck",
            "Az_plus_v",
            "zzAzv",
        ],
    )
    parser.add_argument(
        "--optim_type",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="14,28,47,66,85,105",
        help="Features where directions should be applied for loss calculation. \
                            See oasis_tools.layer_names.LAYER_LOOKUP",
    )
    parser.add_argument("--allow_print", action="store_true", default=False)
    parser.add_argument(
        "--per_layer",
        action="store_true",
        default=False,
        help="If 1 direction should be applied for all layers. \
        Otherwise for each layer a specific direction will be discovered",
    )
    parser.add_argument(
        "--unit_norm",
        action="store_true",
        default=False,
        help="Normalize the directions",
    )
    parser.add_argument(
        "--normalize_images",
        action="store_true",
        default=False,
        help="Applies min-max normalization before computing feature difference ",
    )
    parser.add_argument(
        "--alpha_scaling",
        action="store_true",
        default=False,
        help="If True, applies a linear factor alpha between [-3, 3]",
    )
    parser.add_argument(
        "--alpha_flipping",
        action="store_true",
        default=False,
        help="If True, flips alpha by +-1",
    )
    parser.add_argument(
        "--use_class_mask",
        action="store_true",
        default=False,
        help="If true, spatial sampling. Otherwise global.",
    )
    parser.add_argument(
        "--results_root_dir",
        type=str,
        default="./oasis_results",
        help="Path for saving intermediary and logging results",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="direct",
        help="direct: apply unit norm on parameters, indirect: apply downstream",
    )
    parser.add_argument(
        "--triplet_loss_lambda", type=float, default=1.0, help="weight of triplet loss"
    )
    parser.add_argument(
        "--area_loss_lambda", type=float, default=1.0, help="weight of area loss"
    )
    parser.add_argument(
        "--combined_loss_lambda",
        type=float,
        default=0.0,
        help="weight of combined loss - 0 by default - because not default loss",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument(
        "--not_class_specific",
        action="store_true",
        default=False,
        help="optimizes directions globally, not taking classes into consideration",
    )
    parser.add_argument(
        "--triplet_loss_version", type=str, default="v1", choices=["v1", "v2"]
    )
    parser.add_argument(
        "--max_change_loss_lambda",
        type=float,
        default=1.0,
        help="weight of max_change_loss",
    )
    parser.add_argument("--max_num_classes", type=int, default=5)  
    parser.add_argument(
        "--diff_fct",
        type=str,
        default="l2",
        choices=["l2", "l1", "cos", "pwd1", "pwd2", "cos_orth"],
    )
    parser.add_argument(
        "--diff_fct_in",
        type=str,
        default="l2",
        choices=["l2", "l1", "cos", "pwd1", "pwd2", "cos_orth"],
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="equal",
        choices=["proportional", "reverse_proportional", "equal", "max", "min"],
    )
    parser.add_argument(
        "--loss_format", type=str, default="log", choices=["sum", "ratio", "log"]
    )
    parser.add_argument(
        "--scores", type=str, default="mean"
    )  
    parser.add_argument("--ckpt_every", type=int, default=5)
    parser.add_argument(
        "--latest_ckpt_folder",
        type=str,
        default="/fs/scratch/rng_cr_bcai_dl/jea2rng/tmp_ckpt/",
    )
    parser.add_argument("--direction_model", type=str, default="ours")
    parser.add_argument("--asymmetric_training", action="store_true", default=False)
    parser.add_argument("--sym_loss", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--only_triplet_sym", action="store_true", default=False)
    parser.add_argument("--pixelwise_triplet", action="store_true", default=False)
    parser.add_argument("--quick", action="store_true", default=False)
    parser.add_argument(
        "--optim_classes", type=str, default="2,3,5", help="use 2,3,5 for quick results"
    )
    parser.add_argument("--merge_classes", action="store_true", default=False)
    parser.add_argument("--acc_iter", type=int, default=500)
    parser.add_argument(
        "--stochastic_triplet", action="store_true", default=False
    )  
    parser.add_argument(
        "--mirror_triplet", action="store_true", default=False
    )  
    parser.add_argument(
        "--mirror_lambda", type=float, default=0.1
    )  
    parser.add_argument(
        "--multi_triplet", action="store_true", default=False
    )  
    parser.add_argument(
        "--d_loss", action="store_true", default=False
    )  
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--ortho_loss", action="store_true", default=False)
    parser.add_argument("--d_lambda", type=float, default=0.1)  

    parser.add_argument(
        "--rotation_alpha", type=float, default=None
    )  
    parser.add_argument(
        "--translation_alpha", type=float, default=None
    )  
    parser.add_argument("--zzAzv_alpha", type=float, default=None)
    parser.add_argument("--flip_zzAzv", action="store_true", default=False)

    args = parser.parse_args()

    if args.dataset == "ade20k":
        args.num_classes = 151
        args.aspect_ratio = 1
        args.c = args.num_classes

    elif args.dataset == "cityscapes":
        args.num_classes = 35
        args.aspect_ratio = 2
        args.c = args.num_classes
    elif args.dataset == "celeba":
        args.num_classes = 19
        args.aspect_ratio = 1
        args.c = args.num_classes
    elif args.dataset == "coco":
        args.num_classes = 183
        args.aspect_ratio = 1
        args.c = args.num_classes
    else:
        raise NotImplementedError()

    if args.quick:
        if args.optim_classes is None:
            args.optim_classes = [2, 3, 5]
        else:
            args.optim_classes = [int(c) for c in args.optim_classes.split(",")]
        args.ckpt_every = 1
        args.num_epochs = 5
        args.name += "_quick"
        args.acc_iter = 50

    elif args.optim_classes is not None:
        if args.optim_classes == "all":
            args.optim_classes = [int(c) for c in range(args.num_classes)]
        else:
            args.optim_classes = [int(c) for c in args.optim_classes.split(",")]

    args.scores = args.scores.split("_")

    args.feature_name = []
    features_indices = [
        int(ind) for ind in args.features.split(",") if "vgg" not in ind
    ]
    for index in features_indices:

        args.feature_name.append(LAYER_LOOKUP[index]) 

    if "vgg_penultimate" in args.features:
        args.feature_name.append("vgg_penultimate")
    if "vgg_features" in args.features:
        args.feature_name.append("vgg_features")

    args.avg_l2_norm = (
        torch.linalg.norm(torch.randn(10000, args.z_dim), ord=2, dim=1).mean().item()
    )

    if args.not_class_specific:
        print("overriding args:")
        area_loss_lambda_before = args.area_loss_lambda
        use_class_mask_before = args.use_class_mask
        args.area_loss_lambda = 0.0
        args.use_class_mask = False
        print(f"area_loss_lambda {area_loss_lambda_before} --> {args.area_loss_lambda}")
        print(f"use_class_mask {use_class_mask_before} --> {args.use_class_mask}")
        
        necessary_conditions = (
            args.triplet_loss_lambda > 0 and args.max_change_loss_lambda > 0
        )
        assert necessary_conditions
    else:
        args.max_change_loss_lambda = 0.0

    args.experiment_folder_name = "".join(
        [str(d) for d in list(datetime.now().timetuple())[1:-2]]
    )
    if args.name != "":
        args.experiment_folder_name += "_" + args.name

    args.output_dir = os.path.join(args.results_root_dir, args.experiment_folder_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "args.json"), "w") as file:
        json.dump(args.__dict__, file, sort_keys=True, indent=4)
    
    print(f"<logdir.start>{args.output_dir}<logdir.end>", flush=True)

    return args
