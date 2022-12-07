import argparse 
import torch
from torch.nn import functional as F 
from functools import partial 
from typing import Callable


def _weight_losses(losses: list, sizes: list, weighting_type: str) -> torch.Tensor: 
    """Given a list of loss from different layers,
    compute a weighted sum.

    Args:
        losses (list): List of torch.Tensors holding losses.
        sizes (list): Spatial sizes of the features on which the losses were computed, e.f. [16,32,64,128]
        weighting_type (str): Name of the weighting method.

    Returns:
        torch.Tensor: Weighted sum of losses
    """
    if weighting_type == "equal":
        loss = sum(losses) / len(losses)
    elif weighting_type == "max":
        loss = max(losses) 
    elif weighting_type == "min":
        loss = min(losses) 
    else:
        max_size = max(sizes)
        factors = [s / max_size for s in sizes]
        if weighting_type == "proportional":
            pass
        elif weighting_type == "reverse_proportional":
            factors = factors[::-1]
        loss = sum([f * l for f, l in zip(factors, losses)])
    return loss

def _select_difference_fct(fct_name: str) -> Callable:
    """Returns a function to compute the difference between two features.

    Args:
        fct_name (str): Name of the function

    Returns:
        [type]: a callable function to compute the difference in feature space.
    """
    if fct_name == "l2":
        difference_fct = lambda x1, x2: (x1 - x2).pow(2).sum(1, keepdim=True)
    elif fct_name == "l1":
        difference_fct = lambda x1, x2: torch.abs(x1 - x2).sum(1, keepdim=True)
    elif fct_name == "cos_orth":
        difference_fct = lambda x1, x2: 1 - torch.abs(
            F.cosine_similarity(x1, x2, dim=1)
        )
        
    elif fct_name == "cos":
        difference_fct = lambda x1, x2: 2 - F.cosine_similarity(x1, x2, dim=1)
    elif fct_name == "pwd1":
        difference_fct = partial(_pairwise_channel_dist, p=1.0)
    elif fct_name == "pwd2":
        difference_fct = partial(_pairwise_channel_dist, p=2.0)
    return difference_fct


def compute_max_change_loss(out: dict):
    """This loss is currently not used"""
    max_change_loss = -torch.cdist(out["features"], out["features_shifted"], p=2)
    if "features_shifted_negative" in out:
        max_change_loss = max_change_loss - torch.cdist(
            out["features"], out["features_shifted_negative"], p=2
        )
        max_change_loss = max_change_loss - torch.cdist(
            out["features"], out["features_shifted_positive"], p=2
        )
        max_change_loss = max_change_loss / 3
    max_change_loss = max_change_loss.mean()
    out["max_change_loss"] = max_change_loss.detach() 
    return out, max_change_loss


def compute_combined_loss(
    args: argparse.Namespace,
    label: torch.Tensor,
    out: dict,
    feature_difference_fct: str = "l2",
    weighting_type: str = "equal",
    eps: float = 1e-6,
    feature_difference_fct_inside: Callable = None,
    symmetric_directions = False,
    multi_triplet = False,
    ):

    difference_fct = _select_difference_fct(feature_difference_fct)

    if feature_difference_fct_inside is None:
        difference_fct_inside = difference_fct
    else:
        difference_fct_inside = _select_difference_fct(feature_difference_fct_inside)
    

    chosen_label = out["chosen_label"]
    if not multi_triplet:
        mask = (label.argmax(1, keepdim=True) == chosen_label.view(-1, 1, 1, 1)).float()

    forbidden_keys = ["x", "mask"]

    feature_names = sorted(out["features"].keys())
    feature_list = [out["features"][key] for key in feature_names if not any([key==fkey for fkey in forbidden_keys])]

    feature_names_shifted = sorted(out["features_shifted"].keys())
    feature_list_shifted = [
        out["features_shifted"][key] for key in feature_names_shifted if not any([key==fkey for fkey in forbidden_keys])
    ]

    feature_names_shifted_positive = sorted(out["features_shifted_positive"].keys())
    feature_list_shifted_positive = [
        out["features_shifted_positive"][key] for key in feature_names_shifted_positive if not any([key==fkey for fkey in forbidden_keys])
    ]

    feature_names_shifted_negatve = sorted(out["features_shifted_negative"].keys())
    feature_list_shifted_negatve = [
        out["features_shifted_negative"][key] for key in feature_names_shifted_negatve if not any([key==fkey for fkey in forbidden_keys])
    ]
    if args.mirror_triplet:
        feature_names_shifted_negative_negative = sorted(out["features_shifted_negative_negative"].keys())
        feature_list_shifted_negative_negative = [
            out["features_shifted_negative"][key] for key in feature_names_shifted_negative_negative if not any([key==fkey for fkey in forbidden_keys])
        ]
    else:
        feature_list_shifted_negative_negative = [None]*len(feature_list_shifted_negatve)
    losses = []
    sizes = [] 

    for i, (
        feat,
        feat_shifted,
        feat_shifted_positive,
        feat_shifted_negative, 
        feat_shifted_negative_negative
    ) in enumerate(
        zip(
            feature_list,
            feature_list_shifted,
            feature_list_shifted_positive,
            feature_list_shifted_negatve, 
            feature_list_shifted_negative_negative,
        )
    ):

        B, C, H, W = feat.size()
        sizes.append(H)
        resized_mask = F.interpolate(mask, size=(H, W))

        if args.triplet_loss_version == "v1":
            diff_anchor_negative = difference_fct_inside(feat_shifted, feat_shifted_negative)
            diff_anchor_positive = difference_fct_inside(feat_shifted, feat_shifted_positive)
            if args.mirror_triplet:
                diff_negative_negative = difference_fct_inside(feat_shifted_negative_negative, feat_shifted_negative)

        elif args.triplet_loss_version == "v2":
            diff_anchor_negative = difference_fct_inside(
                feat, feat_shifted_negative - feat_shifted
            )
            diff_anchor_positive = difference_fct_inside(
                feat, feat_shifted_positive - feat_shifted
            )

        if not args.not_class_specific and not args.pixelwise_triplet:
            
            diff_anchor_positive_loss = resized_mask * diff_anchor_positive
            diff_anchor_positive_loss = diff_anchor_positive_loss.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)
            diff_anchor_negative_loss = resized_mask * diff_anchor_negative
            diff_anchor_negative_loss = diff_anchor_negative_loss.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)   
            diff_area_loss = (1-resized_mask) * diff_anchor_negative    
            diff_area_loss = diff_area_loss.sum([2, 3]) / ((-resized_mask).sum([2, 3]) + eps)   
 
            if args.mirror_triplet: 
                diff_negative_negative = resized_mask * diff_negative_negative
                diff_negative_negative = diff_negative_negative.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)   

        trip_pos = diff_anchor_positive_loss*args.beta * args.combined_loss_lambda 
        trip_neg = diff_anchor_negative_loss*args.gamma * args.combined_loss_lambda 
        trip_area = diff_area_loss*args.theta * args.combined_loss_lambda 
        out["trip_pos"] = trip_pos.detach().mean()
        out["trip_neg"] = trip_neg.detach().mean()
        out["trip_area"] = trip_area.detach().mean()
        diff = trip_pos  - trip_neg + trip_area

        if args.mirror_triplet:
            diff = diff - diff_negative_negative * args.mirror_lambda

        if diff.ndim == 3:
            diff = diff.unsqueeze(1)

        if not args.not_class_specific and args.pixelwise_triplet: 
            diff = resized_mask * diff
            diff = diff.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)

            loss = diff.mean()
        else:
            loss = diff.mean()
        losses.append(loss) 
    combi_loss = _weight_losses(losses, sizes, weighting_type)
    out["combi_loss"] = combi_loss.detach() 
    return out, combi_loss


def compute_weighted_triplet_loss(
    args: argparse.Namespace,
    label: torch.Tensor,
    out: dict,
    feature_difference_fct: str = "l2",
    weighting_type: str = "equal",
    eps: float = 1e-6,
    feature_difference_fct_inside: Callable = None,
    symmetric_directions = False,
    multi_triplet = False,
    ):
    """Computes the triplet loss"""

    difference_fct = _select_difference_fct(feature_difference_fct)

    if feature_difference_fct_inside is None:
        difference_fct_inside = difference_fct
    else:
        difference_fct_inside = _select_difference_fct(feature_difference_fct_inside)
    

    chosen_label = out["chosen_label"]
    if not multi_triplet:
        mask = (label.argmax(1, keepdim=True) == chosen_label.view(-1, 1, 1, 1)).float()

    forbidden_keys = ["x", "mask"]

    feature_names = sorted(out["features"].keys())
    feature_list = [out["features"][key] for key in feature_names if not any([key==fkey for fkey in forbidden_keys])]

    feature_names_shifted = sorted(out["features_shifted"].keys())
    feature_list_shifted = [
        out["features_shifted"][key] for key in feature_names_shifted if not any([key==fkey for fkey in forbidden_keys])
    ]

    feature_names_shifted_positive = sorted(out["features_shifted_positive"].keys())
    feature_list_shifted_positive = [
        out["features_shifted_positive"][key] for key in feature_names_shifted_positive if not any([key==fkey for fkey in forbidden_keys])
    ]

    feature_names_shifted_negatve = sorted(out["features_shifted_negative"].keys())
    feature_list_shifted_negatve = [
        out["features_shifted_negative"][key] for key in feature_names_shifted_negatve if not any([key==fkey for fkey in forbidden_keys])
    ]
    if args.mirror_triplet:
        feature_names_shifted_negative_negative = sorted(out["features_shifted_negative_negative"].keys())
        feature_list_shifted_negative_negative = [
            out["features_shifted_negative"][key] for key in feature_names_shifted_negative_negative if not any([key==fkey for fkey in forbidden_keys])
        ]
    else:
        feature_list_shifted_negative_negative = [None]*len(feature_list_shifted_negatve)

    if symmetric_directions:
        assert not multi_triplet, " not impl."
        feature2_names = sorted(out["features2"].keys()) 
        feature2_list = [out["features2"][key] for key in feature2_names if not any([key==fkey for fkey in forbidden_keys])]
        feature_names_shifted_alpha_minus = sorted(out["features_shifted_alpha_minus"].keys()) 
        feature_list_shifted_alpha_minus = [
            out["features_shifted_alpha_minus"][key] for key in feature_names_shifted_alpha_minus if not any([key==fkey for fkey in forbidden_keys])
        ]
        feature_names_shifted_positive_alpha_minus = sorted(out["features_shifted_positive_alpha_minus"].keys()) 
        feature_list_shifted_positive_alpha_minus = [
            out["features_shifted_positive_alpha_minus"][key] for key in feature_names_shifted_positive_alpha_minus if not any([key==fkey for fkey in forbidden_keys])
        ]
        feature_names_shifted_negative_alpha_minus = sorted(out["features_shifted_negative_alpha_minus"].keys()) 
        feature_list_shifted_negative_alpha_minus = [
            out["features_shifted_negative_alpha_minus"][key] for key in feature_names_shifted_negative_alpha_minus if not any([key==fkey for fkey in forbidden_keys])
        ]

    losses = []
    sizes = [] 

    if symmetric_directions:
        assert not multi_triplet, "not impl."
        for i, (
            feat,
            feat_shifted,
            feat_shifted_positive,
            feat_shifted_negative,

            feat2,
            feat_shifted_alpha_minus,
            feat_shifted_positive_alpha_minus,
            feat_shifted_negative_alpha_minus,
        ) in enumerate(
            zip(
                feature_list,
                feature_list_shifted,
                feature_list_shifted_positive,
                feature_list_shifted_negatve,

                feature2_list,
                feature_list_shifted_alpha_minus,
                feature_list_shifted_positive_alpha_minus,
                feature_list_shifted_negative_alpha_minus,
            )
        ):
            B, C, H, W = feat.size()
            sizes.append(H)
            resized_mask = F.interpolate(mask, size=(H, W)) 
            max_pairs = [
                [feat_shifted, feat_shifted_negative],
                [feat_shifted_alpha_minus, feat_shifted_negative_alpha_minus],
                [feat_shifted, feat_shifted_negative_alpha_minus],
                [feat_shifted_alpha_minus, feat_shifted_negative],
                [feat_shifted, feat_shifted_positive_alpha_minus],   
            ]
            min_pairs = [
                [feat_shifted, feat_shifted_positive],
                [feat_shifted_alpha_minus, feat_shifted_positive_alpha_minus],
            ]
            diff = 0
            for max_pair in max_pairs:
                diff_to_max = difference_fct_inside(max_pair[0], max_pair[1])  
                if not args.not_class_specific and not args.pixelwise_triplet:
                    diff_to_max = resized_mask * diff_to_max
                    diff_to_max = diff_to_max.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)
                diff = diff  - diff_to_max

            for min_pair in min_pairs:
                diff_to_min = difference_fct_inside(min_pair[0], min_pair[1])  
                if not args.not_class_specific and not args.pixelwise_triplet:
                    diff_to_min = resized_mask * diff_to_min
                    diff_to_min = diff_to_min.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)
                diff = diff + diff_to_min

            if args.pixelwise_triplet:
                diff = resized_mask * diff
                diff = diff.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)
            loss = diff.mean()
            losses.append(loss) 

    else:
        for i, (
            feat,
            feat_shifted,
            feat_shifted_positive,
            feat_shifted_negative, 
            feat_shifted_negative_negative
        ) in enumerate(
            zip(
                feature_list,
                feature_list_shifted,
                feature_list_shifted_positive,
                feature_list_shifted_negatve, 
                feature_list_shifted_negative_negative,
            )
        ):
            B, C, H, W = feat.size()
            sizes.append(H)

            if not multi_triplet:
                resized_mask = F.interpolate(mask, size=(H, W))

            if args.triplet_loss_version == "v1":
                diff_anchor_negative = difference_fct_inside(feat_shifted, feat_shifted_negative)
                diff_anchor_positive = difference_fct_inside(feat_shifted, feat_shifted_positive)
                if args.mirror_triplet:
                    diff_negative_negative = difference_fct_inside(feat_shifted_negative_negative, feat_shifted_negative)
                     

            elif args.triplet_loss_version == "v2":
                diff_anchor_negative = difference_fct_inside(
                    feat, feat_shifted_negative - feat_shifted
                )
                diff_anchor_positive = difference_fct_inside(
                    feat, feat_shifted_positive - feat_shifted
                )

            if not args.not_class_specific and not args.pixelwise_triplet:
                if not multi_triplet:
                    diff_anchor_positive = resized_mask * diff_anchor_positive
                    diff_anchor_positive = diff_anchor_positive.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)
                else:
                    diff_anchor_positive = diff_anchor_positive.mean(dim=[2,3])

                if not multi_triplet:
                    diff_anchor_negative = resized_mask * diff_anchor_negative
                    diff_anchor_negative = diff_anchor_negative.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)   
                else:
                    diff_anchor_negative = diff_anchor_negative.mean(dim=[2,3])

                if args.mirror_triplet:
                    if not multi_triplet:
                        diff_negative_negative = resized_mask * diff_negative_negative
                        diff_negative_negative = diff_negative_negative.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)   
                    else:
                        diff_negative_negative = diff_negative_negative.mean(dim=[2,3])
                    
                    
            diff = diff_anchor_positive - diff_anchor_negative
            if args.mirror_triplet:
                diff = diff - diff_negative_negative * args.mirror_lambda
            if diff.ndim == 3:
                diff = diff.unsqueeze(1)
            if not args.not_class_specific and args.pixelwise_triplet:
                if not multi_triplet:
                    diff = resized_mask * diff
                    diff = diff.sum([2, 3]) / (resized_mask.sum([2, 3]) + eps)
                loss = diff.mean()
            else:
                loss = diff.mean()
            losses.append(loss) 

    triplet_loss = _weight_losses(losses, sizes, weighting_type)
    out["triplet_loss"] = triplet_loss.detach() 
    return out, triplet_loss


def _masked_texture_similarity(feat_1, feat_2, mask):
    mean_1 = (feat_1 * mask).sum([2, 3], keepdim=True) / (
        mask.sum(dim=[2, 3], keepdim=True) + 1e-6
    )
    mean_2 = (feat_2 * mask).sum([2, 3], keepdim=True) / (
        mask.sum(dim=[2, 3], keepdim=True) + 1e-6
    )
    texture_similarity = (2 * mean_1 * mean_2 + 1e-6) / (
        mean_1 ** 2 + mean_2 ** 2 + 1e-6
    )
    return texture_similarity, mean_1, mean_2


def _masked_structure_similarity(feat_1, feat_2, feat_1_mean, feat_2_mean, mask):

    var_1 = ((feat_1 - feat_1_mean) ** 2) * mask
    var_1 = var_1.sum([2, 3], keepdim=True) / (
        mask.sum(dim=[2, 3], keepdim=True) + 1e-6
    )
    var_2 = ((feat_2 - feat_2_mean) ** 2) * mask
    var_2 = var_2.sum([2, 3], keepdim=True) / (
        mask.sum(dim=[2, 3], keepdim=True) + 1e-6
    )
    cov = (var_1 * var_2) - feat_1_mean * feat_2_mean
    inside_structure_similarity = (2 * cov + 1e-6) / (var_1 + var_2 + 1e-6)
    return inside_structure_similarity


def _pairwise_channel_dist(x1, x2, p: float=2.0):
    B, C, H, W = x1.size()
    x1_reshaped = x1.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
    x2_reshaped = x2.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
    dist = torch.nn.functional.pairwise_distance(x1_reshaped, x2_reshaped, p=p)
    dist = dist.view(B, 1, H, W)
    return dist


def _area_diff(feat, feat_shifted, difference_fct, difference_fct_inside, resized_mask, eps):
    feature_difference_out = difference_fct(feat, feat_shifted)
    feature_difference_in  = difference_fct_inside(feat, feat_shifted)
    masked_feature_difference_inside  = feature_difference_in * resized_mask
    masked_feature_difference_outside = feature_difference_out * (1 - resized_mask)
    diff_inside_mean = torch.sum(masked_feature_difference_inside, dim=[2, 3]) / (
        resized_mask.sum(dim=[2, 3]) + eps
    )
    diff_outside_mean = torch.sum(masked_feature_difference_outside, dim=[2, 3]) / (
        (1 - resized_mask).sum(dim=[2, 3]) + eps
    )
    return diff_inside_mean, diff_outside_mean


def compute_weighted_area_loss(
    args,
    label,
    out,
    feature_difference_fct="l2",  
    weighting_type="equal",  
    loss_format="sum",  
    scores_to_optimize=["mean"],  
    eps=1e-3,
    feature_difference_fct_inside = None,
    symmetric_directions = False,
    multi_triplet = False
):
    """
    scores_to_optimize [list]: list of all scores to optimize, e.g. ["mean", "var", "structure"]
    """
    difference_fct = _select_difference_fct(feature_difference_fct)
    if feature_difference_fct_inside is None:
        difference_fct_inside = difference_fct
    else:
        difference_fct_inside = _select_difference_fct(feature_difference_fct_inside)
    chosen_label = out["chosen_label"]
    mask = (label.argmax(1, keepdim=True) == chosen_label.view(-1, 1, 1, 1)).float()
    out["loss_mask"] = mask
    forbidden_keys = ["x", "mask"]
    feature_names = sorted(out["features"].keys()) 
    feature_list = [out["features"][key] for key in feature_names if not any([key==fkey for fkey in forbidden_keys])]

    if multi_triplet:
        feature_names_shifted = sorted(out["area_features_shifted"].keys()) 
        feature_list_shifted = [
            out["area_features_shifted"][key] for key in feature_names_shifted if not any([key==fkey for fkey in forbidden_keys])
        ]
        
    else:
        feature_names_shifted = sorted(out["features_shifted"].keys()) 
        feature_list_shifted = [
            out["features_shifted"][key] for key in feature_names_shifted if not any([key==fkey for fkey in forbidden_keys])
        ]
    if symmetric_directions:
        feature2_names = sorted(out["features2"].keys()) 
        feature2_list = [out["features2"][key] for key in feature2_names if not any([key==fkey for fkey in forbidden_keys]) ]

        feature_names_shifted_positive = sorted(out["features_shifted_positive"].keys()) 
        feature_list_shifted_positive = [
            out["features_shifted_positive"][key] for key in feature_names_shifted_positive if not any([key==fkey for fkey in forbidden_keys])
        ]

        feature_names_shifted_negative = sorted(out["features_shifted_negative"].keys()) 
        feature_list_shifted_negative = [
            out["features_shifted_negative"][key] for key in feature_names_shifted_negative if not any([key==fkey for fkey in forbidden_keys])
        ]

        feature_names_shifted_positive_alpha_minus = sorted(out["features_shifted_positive_alpha_minus"].keys()) 
        feature_list_shifted_positive_alpha_minus = [
            out["features_shifted_positive_alpha_minus"][key] for key in feature_names_shifted_positive_alpha_minus if not any([key==fkey for fkey in forbidden_keys])
        ]

        feature_names_shifted_negative_alpha_minus = sorted(out["features_shifted_negative_alpha_minus"].keys()) 
        feature_list_shifted_negative_alpha_minus = [
            out["features_shifted_negative_alpha_minus"][key] for key in feature_names_shifted_negative_alpha_minus if not any([key==fkey for fkey in forbidden_keys])
        ]

        feature_names_shifted_alpha_minus = sorted(out["features_shifted_alpha_minus"].keys()) 
        feature_list_shifted_alpha_minus = [
            out["features_shifted_alpha_minus"][key] for key in feature_names_shifted_alpha_minus if not any([key==fkey for fkey in forbidden_keys])
        ]
    
    losses = []
    sizes = []
    if symmetric_directions:
        for i, (feat, feat_shifted, feat2,
            feat_shifted_positive,
            feat_shifted_negative,
            feat_shifted_positive_alpha_minus,
            feat_shifted_negative_alpha_minus,
            feat_shifted_alpha_minus) in enumerate(zip(feature_list, 
            feature_list_shifted, 
            feature2_list,
            feature_list_shifted_positive,
            feature_list_shifted_negative,
            feature_list_shifted_positive_alpha_minus,
            feature_list_shifted_negative_alpha_minus,
            feature_list_shifted_alpha_minus,)):
            B, C, H, W = feat.size()
            sizes.append(H)
            resized_mask = F.interpolate(mask, size=(H, W))
            scores_to_minimize = []
            scores_to_maximize = []

            pairs = [
                [feat, feat_shifted], 
                [feat, feat_shifted_negative],
                [feat2, feat_shifted_positive],
                [feat, feat_shifted_alpha_minus], 
                [feat, feat_shifted_negative_alpha_minus],
                [feat2, feat_shifted_positive_alpha_minus],
                ]

            for pair in pairs:
                diff_inside_mean, diff_outside_mean = _area_diff(pair[0], pair[1], difference_fct, difference_fct_inside, resized_mask, eps) 
                if "mean" in scores_to_optimize:
                    diff_inside_score = diff_inside_mean.mean(dim=1)  
                    diff_outside_score = diff_outside_mean.mean(dim=1)  

                    scores_to_maximize.append(diff_inside_score)
                    scores_to_minimize.append(diff_outside_score)
                else:
                    raise NotImplementedError()

            loss = 0
            for score_max, score_min in zip(scores_to_maximize, scores_to_minimize):
                if loss_format == "sum":
                    loss = loss + score_min - score_max
                elif loss_format == "ratio":
                    loss = loss + score_min / (score_max + eps)
                elif loss_format == "log":
                    loss = loss + torch.log(score_min + eps) - torch.log(score_max + eps)
            loss = loss.mean()
            losses.append(loss)
    else:
        for i, (feat, feat_shifted) in enumerate(zip(feature_list, feature_list_shifted)):
            B, C, H, W = feat.size()
            sizes.append(H)
            resized_mask = F.interpolate(mask, size=(H, W))
            feature_difference_out = difference_fct(feat, feat_shifted)
            feature_difference_in  = difference_fct_inside(feat, feat_shifted)
            masked_feature_difference_inside  = feature_difference_in * resized_mask
            masked_feature_difference_outside = feature_difference_out * (1 - resized_mask)
            diff_inside_mean = torch.sum(masked_feature_difference_inside, dim=[2, 3]) / (
                resized_mask.sum(dim=[2, 3]) + eps
            )
            diff_outside_mean = torch.sum(masked_feature_difference_outside, dim=[2, 3]) / (
                (1 - resized_mask).sum(dim=[2, 3]) + eps
            )
            scores_to_minimize = []
            scores_to_maximize = []
            if "mean" in scores_to_optimize:
                diff_inside_score = diff_inside_mean.mean(dim=1)  
                diff_outside_score = diff_outside_mean.mean(dim=1)  
                scores_to_maximize.append(diff_inside_score)
                scores_to_minimize.append(diff_outside_score)
            if "var" in scores_to_optimize:
                mean_in = diff_inside_mean.view(B, -1, 1, 1)
                mean_out = diff_outside_mean.view(B, -1, 1, 1)
                masked_sqrd_diff_in = ((feature_difference_in - mean_in) ** 2) * resized_mask
                masked_sqrd_diff_out = ((feature_difference_out - mean_out) ** 2) * (
                    1 - resized_mask
                )
                var_in = torch.sum(masked_sqrd_diff_in, dim=[2, 3]) / (
                    resized_mask.sum(dim=[2, 3]) + eps
                )
                var_out = torch.sum(masked_sqrd_diff_out, dim=[2, 3]) / (
                    (1 - resized_mask).sum(dim=[2, 3]) + eps
                )
                diff_inside_var = var_in.mean(dim=1)  
                diff_outside_var = var_out.mean(dim=1)  
                scores_to_maximize.append(diff_inside_var)
                scores_to_minimize.append(diff_outside_var)

            if "structure" in scores_to_optimize:
                (
                    inside_texture_similarity,
                    inside_mean,
                    inside_shifted_mean,
                ) = _masked_texture_similarity(feat, feat_shifted, resized_mask)
                inside_structure_similarity = _masked_structure_similarity(
                    feat, feat_shifted, inside_mean, inside_shifted_mean, resized_mask
                )
                inside_texture_similarity = inside_texture_similarity.squeeze().mean(
                    dim=1
                )  
                inside_structure_similarity = inside_structure_similarity.squeeze().mean(
                    dim=1
                )  
                scores_to_minimize.append(inside_texture_similarity)
                scores_to_minimize.append(inside_structure_similarity)
                (
                    outside_texture_similarity,
                    outside_mean,
                    outside_shifted_mean,
                ) = _masked_texture_similarity(feat, feat_shifted, 1 - resized_mask)
                outside_structure_similarity = _masked_structure_similarity(
                    feat, feat_shifted, outside_mean, outside_shifted_mean, 1 - resized_mask
                )
                outside_texture_similarity = outside_texture_similarity.squeeze().mean(
                    dim=1
                )  
                outside_structure_similarity = outside_structure_similarity.squeeze().mean(
                    dim=1
                )  
                scores_to_maximize.append(outside_texture_similarity)
                scores_to_maximize.append(outside_structure_similarity)
            loss = 0
            for score_max, score_min in zip(scores_to_maximize, scores_to_minimize):
                if loss_format == "sum":
                    loss = loss + score_min - score_max
                elif loss_format == "ratio":
                    loss = loss + score_min / (score_max + eps)
                elif loss_format == "log":
                    loss = loss + torch.log(score_min + eps) - torch.log(score_max + eps)
            loss = loss.mean()
            losses.append(loss)

    final_loss = _weight_losses(losses, sizes, weighting_type)

    out["area_loss"] = final_loss.detach() 
    return out, final_loss
