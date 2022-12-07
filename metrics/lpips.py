from torch import nn
import lpips
from lpips import LPIPS
from torch.nn import functional as F
from utilfunctions import scale_to_plus_min_one


class MaskedLPIPS(nn.Module):
    """A version of lpips where the image difference is only computed inside a binary mask area.
    """
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS(net="vgg").cuda()

    def spatial_average(self, in_tens, keepdim=True):
        return in_tens.mean([2, 3], keepdim=keepdim)

    def masked_spatial_average(self, in_tens, mask, keepdim=True):
        scaled_mask = F.interpolate(mask, size=(in_tens.size(2), in_tens.size(3)))
        in_tens = in_tens * scaled_mask
        spatial_sum = in_tens.sum([2, 3], keepdim=keepdim)
        mask_area = scaled_mask.sum([2, 3], keepdim=keepdim)
        return spatial_sum / (mask_area + 1e-6)

    def upsample(
        self, in_tens, out_HW=(64, 64)
    ):  
        return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)

    def forward(
        self,
        in0,
        in1,
        mask_0=None,
        mask_1=None,
        retPerLayer=False,
        mask_position="feature_space",
        return_as_tensor=False,
    ):
        """
        Modified from. https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py

        If mask_position=="feature_space", mask_0 is multiplied
        with ||feature_1 - feature_2||. In other words, only use
        mask_position=="feature_space" if both images have the
        _same_ label map.

        When in0 and in1 have different label maps, you can use
        mask_position=="input". In this case in0*=mask_0 and
        in1*=mask_1.

        """

        in0 = scale_to_plus_min_one(in0)
        in1 = scale_to_plus_min_one(in1)

        in0_input, in1_input = (
            (self.lpips.scaling_layer(in0), self.lpips.scaling_layer(in1))
            if self.lpips.version == "0.1"
            else (in0, in1)
        )

        if mask_position == "input":
            in0 = in0 * mask_0
            in1 = in1 * mask_1

        outs0, outs1 = self.lpips.net.forward(in0_input), self.lpips.net.forward(
            in1_input
        )
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.lpips.L):
            feats0[kk] = lpips.normalize_tensor(outs0[kk])
            feats1[kk] = lpips.normalize_tensor(outs1[kk])

            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips.lpips:

            if self.lpips.spatial:
                res = [
                    self.upsample(self.lpips.lins[kk](diffs[kk]), out_HW=in0.shape[2:])
                    for kk in range(self.lpips.L)
                ]
            else:
                
                if mask_position == "feature_space":
                    res = [
                        self.masked_spatial_average(
                            self.lpips.lins[kk](diffs[kk]), mask_0, keepdim=True
                        )
                        for kk in range(self.lpips.L)
                    ]
                else:
                    res = [
                        self.spatial_average(
                            self.lpips.lins[kk](diffs[kk]), keepdim=True
                        )
                        for kk in range(self.lpips.L)
                    ]
        else:
            print("dont use this")
            if self.lpips.spatial:
                res = [
                    self.upsample(
                        diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]
                    )
                    for kk in range(self.lpips.L)
                ]
            else:
                input_to_average = diffs[kk].sum(dim=1, keepdim=True) 
                res = [
                    self.spatial_average(input_to_average, keepdim=True)
                    for kk in range(self.lpips.L)
                ]

        val = 0
        for l in range(self.lpips.L):
            val += res[l]

        if not return_as_tensor and not retPerLayer:
            val = val.item()

        if retPerLayer:
            return (val, res)
        else:
            return val
