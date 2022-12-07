
from cv2 import normalize
from pytorch_msssim import ms_ssim
import torch
from utilfunctions import min_max_scaling
from typing import Union


def masked_msssim(image_1: torch.Tensor, 
                 image_2: torch.Tensor, 
                 mask_1: torch.Tensor, 
                 mask_2: torch.Tensor, 
                 return_as_tensor: bool = False) -> Union[torch.Tensor, float]:

    """ Returns ms-ssim between two masked images. Applies the binary mask before computing ms-ssim.
    
    Assumes image_1 and image_2 have the same binary_mask (same label map)!

    Args:
        image_1 (torch.Tensor): Image 1
        image_2 (torch.Tensor): Image 2
        mask_1 (torch.Tensor): mask 1
        mask_2 (torch.Tensor): mask 2
        return_as_tensor ([type], optional): Returns ms-ssim as tensor instead of float. Defaults to bool:False.

    Returns:
        Union[torch.Tensor, float]: [description]
    """
    binary_mask_1 = mask_1.float()
    binary_mask_2 = mask_2.float()

    msssim = ms_ssim(
        min_max_scaling(image_1) * binary_mask_1,
        min_max_scaling(image_2) * binary_mask_2,
        data_range=1,
        size_average=False,
    )
    if return_as_tensor:
        return msssim
    else:
        return msssim.item()

