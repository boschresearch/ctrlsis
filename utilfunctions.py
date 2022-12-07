import argparse
from numpy import ndim
import torch
from torch import nn
import pandas as pd
import random 
import os 
from torch.distributions.categorical import Categorical  
import subprocess

from oasis_tools.utils import label_to_one_hot
from torch.utils.data.dataloader import DataLoader

import torch
from torch.nn import functional as F 
from collections.abc import Mapping
import copy 

def accumulate_standing_stats(
    main_args, generator, dataloader, device="cuda:0", max_iter=None
):

    generator.train()
    tmp_dataloader = copy.deepcopy(dataloader)
    print("accumulating standing stats")

    for i, batch in enumerate(tmp_dataloader):
        if i % 50 == 0:
            if max_iter is not None:
                print(
                    f"acc progress: {100*i/max_iter} % (batch {i}/{max_iter} of bs {batch['image'].size(0)})"
                )
            else:
                print(
                    f"acc progress: {100*i/len(tmp_dataloader)} % (batch {i}/{len(tmp_dataloader)} of bs {batch['image'].size(0)})"
                )

        _, label = batch["image"].to(device), batch["label"].to(device)

        if label.ndim == 3:
            label = label.unsqueeze(1)
        label = label_to_one_hot(
            label.long(), num_classes=main_args.num_classes
        )  
        
        out = generator(label)
        
        if max_iter is not None:
            if i >= max_iter:
                break
    return generator

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

AVG_L2_NORM = torch.linalg.norm(torch.randn(10000, 64), ord=2, dim=1).mean().item()
AVG_L2_NORM_STD = torch.linalg.norm(torch.randn(10000, 64), ord=2, dim=1).std().item()

def fid_generator_wrapper(
    generator,
    mode,
    direction_model=None,
    max_k=5,
    alpha=None,
    global_shift=False,
    rand_k_per_class=False,
    rand_alpha_flip=False,
    return_mask=False,
    z_dim=64,
    alpha_scale_mode="default",
    use_matrix = False,
):

    if alpha is None:
        alpha = AVG_L2_NORM

    if mode == "spade_vanilla":

        def wrapper(label):
            z = torch.randn(label.size(0), z_dim, device=label.device)
            return generator(label, z)

    elif mode == "shifted":

        return_keys = ["image_shifted", "image"]
        if return_mask:
            return_keys += ["noise_mask"]

        if global_shift:
            print("GLOBAL SHIFT")
            print("rand_k_per_class", rand_k_per_class)
            assert not use_matrix, "not implemented"

            def wrapper(label):  
                tmp_alpha = alpha
                if alpha_scale_mode == "3sigma":
                    tmp_alpha = (
                        AVG_L2_NORM
                        + 2
                        * AVG_L2_NORM_STD
                        * torch.rand((label.size(0),), device=label.device)
                        - AVG_L2_NORM_STD
                    )
                elif alpha_scale_mode == "zeroeight":
                    tmp_alpha = AVG_L2_NORM * torch.rand(
                        (label.size(0),), device=label.device
                    )

                if rand_alpha_flip:
                    tmp_alpha = tmp_alpha * (
                        torch.randint(2, (label.size(0),), device=label.device) * 2 - 1
                    )
                else:
                    tmp_alpha = tmp_alpha

                with torch.no_grad():
                    z_batch = torch.randn(label.size(0), z_dim, device=label.device)

                    direction = direction_model.get_random_directions_3d(
                        label_map=label,
                        alpha=1.0,
                        z=z_batch,
                        c=None,
                        use_all_labels=True,
                        rand_k_per_class=rand_k_per_class,
                    )  
                    
                    out = apply_shift(
                        z=z_batch,
                        direction=direction,
                        label=label,
                        image_generator=generator,
                        chosen_label=None,
                        num_layers=8,
                        use_class_mask=True,
                        alpha=tmp_alpha,
                        return_keys=return_keys,
                    )
                if return_mask:
                    return out["image_shifted"], out["noise_mask"]
                else:
                    return out["image_shifted"]

        else:
            print("LOCAL SHIFT")
            k_distr = Categorical(probs=torch.ones((32, max_k)) / max_k)

            def wrapper(label):  
                tmp_alpha = alpha
                if alpha_scale_mode == "3sigma":
                    tmp_alpha = (
                        AVG_L2_NORM
                        + 2
                        * AVG_L2_NORM_STD
                        * torch.rand((label.size(0),), device=label.device)
                        - AVG_L2_NORM_STD
                    )
                elif alpha_scale_mode == "zeroeight":
                    tmp_alpha = AVG_L2_NORM * torch.rand(
                        (label.size(0),), device=label.device
                    )

                if rand_alpha_flip:
                    tmp_alpha = tmp_alpha * (
                        torch.randint(2, (label.size(0),), device=label.device) * 2 - 1
                    )
                else:
                    tmp_alpha = tmp_alpha

                with torch.no_grad():
                    z_batch = torch.randn(label.size(0), z_dim, device=label.device)

                    k_tensor = k_distr.sample()[: label.size(0)].to(label.device)
                    class_tensor = pick_labels(label)
                    class_tensor = class_tensor.to(label.device)
                    
                    direction = direction_model.get_direction_batched(
                        k=k_tensor, c=class_tensor, alpha=1.0, z=z_batch, return_matrix = use_matrix,
                    )
                    
                    if use_matrix:
                        direction, matrix = direction["v"], direction["A"]
                        
                        matrix = matrix.to(label.device)
                    else:
                        matrix = None

                    direction = direction.to(label.device)

                    out = apply_shift(
                        z=z_batch,
                        direction=direction,
                        label=label,
                        image_generator=generator,
                        chosen_label=class_tensor,
                        num_layers=7,
                        use_class_mask=True,
                        alpha=tmp_alpha,
                        return_keys=return_keys,
                        matrix=matrix,
                        rotation_alpha=1.0,
                        zzAzv_alpha=tmp_alpha,
                    ) 

                if return_mask:
                    return out["image_shifted"], out["noise_mask"]
                else:
                    return out["image_shifted"]

    return wrapper

def get_full_file_paths_recursively(
    path, include_folder_and_file_patterns=[], exclude_folder_and_file_patterns=[]
):
    file_list = []
    contents = os.listdir(path)
    contents = [os.path.join(path, c) for c in contents]

    for i in range(len(exclude_folder_and_file_patterns)):
        if exclude_folder_and_file_patterns[i][-1] == "/":
            exclude_folder_and_file_patterns[i] = exclude_folder_and_file_patterns[i][
                :-1
            ]

    contents = [
        c
        for c in contents
        if not any([exc in c for exc in exclude_folder_and_file_patterns])
    ]

    for c in contents:
        if not any([exc in c for exc in exclude_folder_and_file_patterns]):

            if os.path.isdir(c):
                file_list += get_full_file_paths_recursively(
                    c,
                    include_folder_and_file_patterns,
                    exclude_folder_and_file_patterns,
                )
            else:
                
                if include_folder_and_file_patterns == [] or any(
                    [inc in c for inc in include_folder_and_file_patterns]
                ):
                    file_list.append(c)
    return file_list

def random_string(length=5):
    return "".join(
        [random.choice("abcdefghijklmnopqrstuvwxyz1234567890") for _ in range(length)]
    )

def preprocess_oasis_input(opt, data):
    data["label"] = data["label"].long()
    if opt.gpu_ids != "-1":
        data["label"] = data["label"].cuda()
        data["image"] = data["image"].cuda()
    label_map = data["label"]
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data["image"], input_semantics

def preprocess_oasis_input_v2(data, gpu_ids="0", semantic_nc=151):
    data["label"] = data["label"].long()
    if gpu_ids != "-1":
        data["label"] = data["label"].cuda()
        data["image"] = data["image"].cuda()
    label_map = data["label"]
    bs, _, h, w = label_map.size()
    nc = semantic_nc
    if gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data["image"], input_semantics

def deep_dict_update(target: dict, source: dict) -> dict:
    """
    Equivalent to target.update(source), but for
    nested dictionaries. Modifies target in place.
    """
    for key, value in source.items():
        if isinstance(value, Mapping) and value:
            returned = deep_dict_update(target.get(key, {}), value)
            target[key] = returned
        else:
            target[key] = source[key]
    return target

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def _load_chardict():
    import numpy as np
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    height = 200
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*() _-+=[]{}':\\/<>.,?~"
    # download some font
    # https://www.fontsquirrel.com/fonts/list/classification/monospaced
    # (should be monospaced)
    font = ImageFont.truetype(
        "Anonymous.ttf", height)
    char_dict_numpy = {}
    char_dict_torch = {}

    for character in chars:
        img = Image.fromarray(
            np.uint8(np.zeros((height, int(height * 0.6), 3))))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), character, (255, 0, 0), font=font)
        letter = np.asarray(img)
        char_dict_numpy[character] = letter
        char_dict_torch[character] = (
            torch.from_numpy(letter).permute(
                2, 0, 1) > 0).float()

    return char_dict_torch

#char_dict_torch = torch.load("assets/char_dict_torch.pth")

def write_on_image(
    text: str,
    image_tensor: torch.Tensor,
    font_pixel_height=20,
    color="white",
    first_batch_element_only=False,
    transparent=True,
) -> torch.Tensor:
    """Writes text on an image.

    Args:
        text (str): Text to write.
        image_tensor (torch.Tensor): Rgb tensor of size B, 3, H, W
        font_pixel_height (int, optional): Size of letters in pixels. Defaults to 20.
        color (str, optional): Color of the text. Defaults to "white".
        first_batch_element_only (bool, optional): Only writes on first image in the batch. Defaults to False.
        transparent (bool, optional): Make background of text field transparent. Defaults to True.

    Returns:
        torch.Tensor: [description]
    """
    char_dict_torch = _load_chardict()

    text_tensor = torch.cat(
        [char_dict_torch[character].to(image_tensor.device) for character in text],
        dim=2,
    )

    ch, h, w = text_tensor.size()
    B, C, H, W = image_tensor.size()

    text_tensor = text_tensor.unsqueeze(0)
    new_h, new_w = font_pixel_height, int(font_pixel_height / h * w)
    text_tensor = F.interpolate(text_tensor, size=(new_h, new_w), mode="nearest")

    text_mask = text_tensor.sum(1, keepdim=True)
    text_tensor = torch.cat([text_mask, text_mask, text_mask], dim=1)
    if color == "white" or color == "black":
        pass
    elif color == "blue":
        text_tensor[:, 0, :, :] = 0
        text_tensor[:, 1, :, :] = 0
    elif color == "green":
        text_tensor[:, 0, :, :] = 0
        text_tensor[:, 2, :, :] = 0
    elif color == "red":
        text_tensor[:, 1, :, :] = 0
        text_tensor[:, 2, :, :] = 0

    image_tensor_with_text = image_tensor.clone()

    text_tensor = text_tensor[:, :, :H, :W]

    if first_batch_element_only:
        if transparent:
            image_tensor_with_text[0, :, :new_h, :new_w] = (
                1 - text_mask
            ) * image_tensor_with_text[
                0, :, :new_h, :new_w
            ] + text_mask * text_tensor * (
                0 if color == "black" else 1
            )
        else:
            image_tensor_with_text[0, :, :new_h, :new_w] = (
                1 - text_tensor if color == "black" else text_tensor
            )
    else:
        if transparent:
            image_tensor_with_text[:, :, :new_h, :new_w] = (
                1 - text_mask
            ) * image_tensor_with_text[
                :, :, :new_h, :new_w
            ] + text_mask * text_tensor * (
                0 if color == "black" else 1
            )
        else:
            image_tensor_with_text[:, :, :new_h, :new_w] = (
                1 - text_tensor if color == "black" else text_tensor
            )
    return image_tensor_with_text

top_c = None

def get_top_c_classes(c: int) -> list:
    """Return top c biggest classes

    Args:
        c (int): return top c classes by area.

    Returns:
        list: Returns c classes with largest total area in the training set.
    """
    global top_c
    if top_c is None:
        top_c = pd.read_csv("notebooks/AvgPixelsPerClass.csv")["class"][0:c].values
    return top_c

def _select_random_unique_value(tensor: torch.Tensor) -> torch.Tensor:
    """returns a randomly chosen value of all unique values"""
    unique = tensor.unique()
    if 0 in unique and len(unique) > 1:
        unique = unique[1:]
    value = unique[torch.randperm(unique.numel())[0]]
    return value

def _select_topc_value(tensor: torch.Tensor, c: int) -> torch.Tensor:
    """Given a label map of size (B,1,H,W), return a label from this map
        which is contained in the top c classes, by object area. If no such
        class is in the label map, choose one at random.

    Args:
        tensor (torch.Tensor): Label map
        c (int): Number of top classes

    Returns:
        torch.Tensor: Chosen class
    """
    unique = tensor.unique()
    topc = get_top_c_classes(c)
    if 0 in unique and len(unique) > 1:
        unique = unique[1:]
    unique_set = set(unique.cpu().numpy())
    values = unique_set.intersection(set(topc))
    if len(values) == 0:
        
        return _select_random_unique_value(tensor)
    else:
        value = random.choice(list(values))
        return torch.tensor(value)

def pick_labels(label, c=None, optim_classes=None):
    """Picks a random label from a semantic label map on the batch level

    Args:
        label (torch.Tensor): semantic label map
        c (int, optional): Just consider top c classes by average pixel count on dataset. Defaults to None.
        optim_classes (list): list of int = _only_ classes to consider for optim

    Returns:
        torch.Tensor: tensor with chosen label as one-hot vector
    """
    if label.ndim == 4 and label.size(1) > 1:
        index_map = label.argmax(1)
    else:
        index_map = label
    batch_size = index_map.size(0)
    chosen_labels = torch.zeros(batch_size)
    for b in range(batch_size):
        if optim_classes is None:
            if c is None:
                chosen_labels[b] = _select_random_unique_value(index_map[b])
            else:
                chosen_labels[b] = _select_topc_value(index_map[b], c)

        else:
            
            if len(optim_classes) > 1:
                unique = index_map[b].unique()
                
                choices = set(unique.cpu().numpy()).intersection(
                    set(optim_classes.cpu().numpy())
                )
                
                chosen_labels[b] = random.sample(choices, 1)[0]
                
            else:
                chosen_labels[b] = optim_classes[0]
            
    return chosen_labels

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def min_max_scaling(image: torch.Tensor) -> torch.Tensor:
    """Scale all values between 0 and 1.

    Args:
        image (torch.Tensor): Any tensor.

    Returns:
        torch.Tensor: Scaled tensor.
    """
    return (image - image.min()) / (image.max() - image.min())

def scale_to_plus_min_one(image: torch.Tensor) -> torch.Tensor:
    """Scale tensor values between -1 and +1

    Args:
        image (torch.Tensor): Any torch tensor.

    Returns:
        torch.Tensor: Scaled image.
    """
    return min_max_scaling(image) * 2 - 1

def get_label_maps(
    args: argparse.Namespace, dataloader: DataLoader, list_of_images: list
) -> torch.Tensor:
    """Given a list of image names (without file ending), return a batch
    of the corresponding label maps of these images.

    Args:
        args (argparse.Namespace): Parsed arguments.
        dataloader (torch.utils.data.dataloader.DataLoader): Dataloader.
        list_of_images (list): List of strings of image names.

    Returns:
        torch.Tensor: Batch of size (batch_size, num_classes, height, width)
    """

    saved_labelmaps = {}
    for img_i, img in enumerate(list_of_images):
        if img not in saved_labelmaps:

            load_image_wo_ending = img.split(".")[0]
            if args.dataset == "coco":
                batch_of_size_one = dataloader.dataset.get_single_label(
                    load_image_wo_ending
                )
            else:
                batch_of_size_one = dataloader.dataset.get_single_label_fast(
                    load_image_wo_ending
                )
            label = batch_of_size_one["label"]

            if img_i == 0:
                labelmap_batch = torch.empty(
                    len(list_of_images), 1, label.size()[-2], label.size()[-1]
                )  

            labelmap_batch[img_i] = label  
            saved_labelmaps[img] = label
        else:
            
            labelmap_batch[img_i] = saved_labelmaps[img]

    if labelmap_batch.ndim == 3:
        labelmap_batch = labelmap_batch.unsqueeze(1)

    labelmap_batch = labelmap_batch.to(args.device)
    labelmap_batch = label_to_one_hot(
        labelmap_batch.long(), num_classes=args.num_classes
    )  
    return labelmap_batch

def _adjust_vector_length(direction: torch.Tensor, alpha: float) -> torch.Tensor:
    """Normalize input vectors to have l2 norm equal to alpha.

    Args:
        direction (torch.Tensor): Tensor of size (batch size, num generator layers, noise dimension)
        alpha (float): Length (l2 norm) to which directions are scaled.

    Returns:
        torch.Tensor: The input tensor, with adjusted l2 norm.
    """
    assert direction.ndim == 3 or direction.ndim == 5
    
    len_dir = torch.linalg.norm(direction, ord=2, dim=2, keepdim=True)

    if isinstance(alpha, torch.Tensor):
        if direction.ndim == 3:
            tmp_alpha = alpha.view(alpha.size(0), 1, 1)
        elif direction.ndim == 5:
            tmp_alpha = alpha.view(alpha.size(0), 1, 1, 1, 1)
    else:
        tmp_alpha = alpha

    direction = direction / len_dir * tmp_alpha

    return direction

def apply_shift(
    z: torch.Tensor,
    direction: torch.Tensor,
    label: torch.Tensor,
    image_generator: nn.Module,
    chosen_label: torch.Tensor = None,
    num_layers: int = 7,
    use_class_mask: bool = True,
    alpha: float = 1.0,
    return_keys: list = [
        "image",
        "image_shifted",
        "image_shifted_no_mask",
        "noise_mask",
        
    ],
    specific_return_features=None,
    
    matrix=None,
    rotation_alpha=None,
    zzAzv_alpha=None,
) -> dict:  
    """Given intitial noise, an image generator, and latent directions, generate
     images from the shifted noise.

    Args:
        z (torch.Tensor): Initial noise of size (batch size, noise dim)
        direction (torch.Tensor): Directions of size (batch size, num generator blocks, noise dim)
        label (torch.Tensor): Label map of size (batch size, num classes, height, width)
        image_generator (nn.Module): An nn.Module with a forward function, transforming noise to images.
        chosen_label (torch.Tensor, optional): Labels to which apply noise shift. Tensor of size (batch size,). Defaults to None.
        num_layers (int, optional): Number of image generator blocks that can take noise as input. Defaults to 7.
        use_class_mask (bool, optional): Apply shifted noise with binary mask for chosen_label. Defaults to True.
        alpha (float, optional): Length to scale directions. Defaults to 1.0.
        return_keys (list, optional): List of desired outputs. Defaults to ["image", 'image_shifted','image_shifted_no_mask', 'noise_mask'].

    Returns:
        [dict]: Dict. The keys are the strings in 'return_keys'. The values are the corresponding generated image tensors.
    """
    if rotation_alpha is None:
        rotation_alpha = 1.0

    batch_size = label.size(0)
    H = label.size(2)
    W = label.size(3)
    z_dim = z.size(1)

    if direction.ndim == 1:
        direction = direction.unsqueeze(0)

    use_one_direction_for_one_class = direction.ndim == 2
    use_many_directions_for_many_classes = direction.ndim == 4

    if direction.size(1) != 64:
        direction = direction.view(direction.size(0), num_layers, 64)
    if direction.ndim == 2:
        direction = direction.unsqueeze(1)

    if use_one_direction_for_one_class:
        
        direction = direction.expand(batch_size, num_layers, -1)
        
        per_layer_noise_original = z.unsqueeze(1).expand(-1, num_layers, -1)

    elif use_many_directions_for_many_classes:
        assert matrix is None, "not implemented yet"
        
        direction = direction.expand(batch_size, num_layers, -1, -1, -1)
        
        per_layer_noise_original = z.view(batch_size, 1, z_dim, 1, 1)
        per_layer_noise_original = per_layer_noise_original.expand(
            -1, -1, -1, H, W
        )  
        per_layer_noise_original = per_layer_noise_original.expand(
            -1, num_layers, -1, -1, -1
        )  
    else:
        raise ValueError("Dimension of direction must be 2 or 5")

    direction = _adjust_vector_length(direction, alpha)

    if matrix is None:
        per_layer_noise_shifted = per_layer_noise_original + direction  
        
    else:
        matrix = matrix.view(batch_size, 1, z_dim, z_dim)
        matrix = matrix.expand(batch_size, num_layers, -1, -1).contiguous()

        num_b, num_layer, z_dim = per_layer_noise_original.size()
        per_layer_noise_affine = torch.bmm(
            rotation_alpha * matrix.view(num_b * num_layer, z_dim, z_dim),
            per_layer_noise_original.reshape(num_b * num_layer, z_dim, 1),
        )
        per_layer_noise_affine = per_layer_noise_affine.view(num_b, num_layer, z_dim)

        per_layer_noise_shifted = per_layer_noise_affine + direction

        if zzAzv_alpha is not None:
            
            per_layer_noise_shifted = _adjust_vector_length(
                per_layer_noise_shifted, 1.0
            )
            
            if torch.is_tensor(zzAzv_alpha):
                zzAzv_alpha = zzAzv_alpha.view(per_layer_noise_original.size(0),1,1)

            per_layer_noise_shifted = (
                per_layer_noise_original + zzAzv_alpha * per_layer_noise_shifted
            )

    outputs = {}

    if "image" in return_keys or "features" in return_keys:
        return_layers = ["x"]
        if "features" in return_keys:
            return_layers += specific_return_features

        features = image_generator(
            label, z, return_layers=["x"], per_layer_noise=per_layer_noise_original
        )
        outputs["image"] = features["x"]

        if "features" in return_keys:
            outputs["features"] = {
                key: value
                for key, value in features.items()
                if (key != "mask" and key != "x")
            }

    if "image_shifted_no_mask" in return_keys:
        if use_many_directions_for_many_classes:
            
            outputs["image_shifted_no_mask"] = None
        else:
            return_layers = ["x"]
            if "features_shifted_no_mask" in return_keys:
                return_layers += specific_return_features

            features = image_generator(
                label,
                z + direction[:, 0],
                return_layers=return_layers,
            )
            outputs["image_shifted_no_mask"] = features["x"]
            if "features_shifted_no_mask" in return_keys:
                outputs["features_shifted_no_mask"] = {
                    key: value
                    for key, value in features.items()
                    if (key != "mask" and key != "x")
                }

    if "image_shifted" in return_keys or "features_shifted" in return_keys:
        return_layers = ["x"]
        if "features_shifted" in return_keys:
            return_layers += specific_return_features

        features_shifted = image_generator(
            label,
            z,
            return_layers=return_layers,
            per_layer_noise=per_layer_noise_shifted,
            label_to_shift=chosen_label if use_class_mask else None,
        )
        outputs["image_shifted"] = features_shifted["x"]

        if "features_shifted" in return_keys:
            outputs["features_shifted"] = {
                key: value
                for key, value in features_shifted.items()
                if (key != "mask" and key != "x")
            }

        if "mask" in features_shifted and "noise_mask" in return_keys:
            
            outputs["noise_mask"] = features_shifted["mask"]

    return outputs

