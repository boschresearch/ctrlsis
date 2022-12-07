"""
Here we define a "transparent" OASIS generator, which can be used as a 
drop-in replacement for the normal OASIS_Generator class.
Transparent means, it allows to return every intermediate result. 

LIST OF LAYERS: oasis_tools.layer_names

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from oasis.models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F 


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == "instance":
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == "syncbatch":
        return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == "batch":
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError(
            "%s is not a recognized param-free norm type in SPADE" % opt.param_free_norm
        )


class SPADE(nn.Module):
    """
    This clone of the SPADE class is changed such that it returns all intermediate outputs.
    """

    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta

        return normalized, actv, gamma, beta, out


class ResnetBlock_with_SPADE(nn.Module):
    """
    This clone of the ResnetBlock_with_SPADE class is changed such that it returns all intermediate outputs.
    """

    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        sp_norm = get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            normalized_s, actv_s, gamma_s, beta_s, norm_s = self.norm_s(x, seg)
            x_s = self.conv_s(norm_s)
        else:
            x_s = x

        # First spade norm
        normalized_0, actv_0, gamma_0, beta_0, norm_0 = self.norm_0(x, seg)
        dx1 = self.conv_0(self.activ(norm_0))

        # Second spade norm
        normalized_1, actv_1, gamma_1, beta_1, norm_1 = self.norm_1(dx1, seg)
        dx2 = self.conv_1(self.activ(norm_1))

        out = x_s + dx2
        ###############################################################

        output_dict = {}
        if self.learned_shortcut:
            output_dict["shortcut_spade_norm_layer_after_batchnorm"] = normalized_s
            output_dict["shortcut_spade_norm_layer_output_of_mlp_shared"] = actv_s
            output_dict["shortcut_spade_norm_layer_output_of_gamma"] = gamma_s
            output_dict["shortcut_spade_norm_layer_output_of_beta"] = beta_s
            output_dict["shortcut_spade_norm_layer_output"] = norm_s

        # First spade norm
        output_dict["conv_on_shortcut_spade_norm_layer"] = x_s
        output_dict["first_spade_norm_layer_after_batchnorm"] = normalized_0
        output_dict["first_spade_norm_layer_output_of_mlp_shared"] = actv_0
        output_dict["first_spade_norm_layer_output_of_gamma"] = gamma_0
        output_dict["first_spade_norm_layer_output_of_beta"] = beta_0
        output_dict["first_spade_norm_layer_output"] = norm_0
        output_dict["conv_on_first_spade_norm_layer"] = dx1

        # Second spade norm
        output_dict["second_spade_norm_layer_after_batchnorm"] = normalized_1
        output_dict["second_spade_norm_layer_output_of_mlp_shared"] = actv_1
        output_dict["second_spade_norm_layer_output_of_gamma"] = gamma_1
        output_dict["second_spade_norm_layer_output_of_beta"] = beta_1
        output_dict["second_spade_norm_layer_output"] = norm_1
        output_dict["conv_on_second_spade_norm_layer"] = dx2
        output_dict["resnet_block_output"] = out

        return output_dict


class OASIS_Generator(nn.Module):
    """
    This class is identical to the original OASIS_Generator class,
    except that all intermediate outputs are saved in
    a variable named "intermediate_features_dict". For this, a
    couple of things had to be rewritten.

    However, the inputs and output of this class are still the same
    as in the original OASIS_Generator class,
    except if the return_layers argument in the forward function is used.
    In this case, the forward function returns a dict
    of the intermediate outputs.

    For an overiew of all layers, see LIST OF LAYERS at the top of this file.

    example:
    return_layers = [
        "block_0_first_spade_norm_layer_output_of_mlp_shared",
        "block_1_first_spade_norm_layer_output_of_mlp_shared",
    ]
    generator = GeneratorWrapper(opt)
    out = generator(label, z, return_layers)


    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        ch = opt.channels_G

        self.channels = [16 * ch, 16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch, 1 * ch]
        self.channels = self.channels[-opt.num_res_blocks - 1 :]

        self.init_W, self.init_H = self.compute_latent_vector_size(opt)

        self.conv_img = nn.Conv2d(self.channels[-1], opt.img_channels, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])

        for i in range(len(self.channels) - 1):
            self.body.append(
                ResnetBlock_with_SPADE(self.channels[i], self.channels[i + 1], opt)
            )

        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(
                self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1
            )
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)
 
    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2 ** (opt.num_res_blocks - 1))
        h = round(w / opt.aspect_ratio)

        return h, w

    # profile
    def forward(
        self,
        input,
        z=None,
        return_layers=None,
        shifted_noise_injection_layers=[],
        per_layer_noise=None,
        label_to_shift=None,
    ):
        """
        input: label map of size (batch size, num classes, height, width)
        z: gaussian noise of size (batch size, z dim)
        shifted_noise_injection_layers: list of layers to inject in range [-1,..,5]
        per_layer_noise:
        - if 1D noise -> tensor of size (batch size, num layers, z dim)
        - if 3D noise -> tensor of size (batch size, num layers, z dim, height, width)
        label_to_shift: tensor of size (batch_size) and type int. ignored if per_layer_noise = 3D noise
        """
        if per_layer_noise is not None:
            if per_layer_noise.ndim == 5:
                assert (
                    label_to_shift is None
                ), "You should apply the label specific shift when you make the 3D direction tensor"

        intermediate_features_dict = {}
        batch_size = input.size(0)

        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if label_to_shift is not None:
            mask = (
                input.argmax(dim=1, keepdim=True)
                == label_to_shift.view(batch_size, 1, 1, 1)
            ).float()

        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"

            if z is None:
                z = torch.randn(
                    seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev
                )

            if per_layer_noise is not None:
                z_layer = per_layer_noise[:, 0]
                if per_layer_noise.ndim == 3:
                    # make shifted 3D noise
                    z_layer = z_layer.view(z_layer.size(0), self.opt.z_dim, 1, 1)
                    z_layer = z_layer.expand(
                        z_layer.size(0), self.opt.z_dim, seg.size(2), seg.size(3)
                    )

            if z is not None and z.ndim == 2:
                # if noise is in format (batch_size, z_dim)
                # make 3D noise
                z = z.view(z.size(0), self.opt.z_dim, 1, 1)
                z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            else:
                # if noise is in format (batch_size, z_dim, H, W)
                z = z.to(dev)

            if per_layer_noise is None:
                # if no shift,
                # take default 3D noise as noise for the input layer
                z_layer = z
            else:
                if label_to_shift is not None:
                    # recursive_check(z, 'tg: z')
                    z_layer = mask * z_layer + (1 - mask) * z
                    # recursive_check(z_layer, 'z_layer')

            seg = torch.cat((z_layer, seg), dim=1)

        elif zseg_tensor is not None:
            seg = zseg_tensor

        if shifted_noise_injection_layers != []:
            raise NotImplementedError("This is zombi code that may be revived")
            z_layer = z_shifted.view(z_shifted.size(0), self.opt.z_dim, 1, 1)
            z_layer = z_layer.expand(
                z_layer.size(0), self.opt.z_dim, input.size(2), input.size(3)
            )  # (B,64,H,W)
            seg_shifted = torch.cat((z_layer, input), dim=1)

        if -1 in shifted_noise_injection_layers:
            raise NotImplementedError("This is zombi code that may be revived")
            x = F.interpolate(seg_shifted, size=(self.init_W, self.init_H))
        else:
            x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        intermediate_features_dict["fc"] = x

        for i in range(self.opt.num_res_blocks):

            if per_layer_noise is not None:
                z_layer = per_layer_noise[:, i + 1]
                if per_layer_noise.ndim == 3:
                    # scale to 3D if not already
                    z_layer = z_layer.view(z_layer.size(0), self.opt.z_dim, 1, 1)
                    z_layer = z_layer.expand(
                        z_layer.size(0), self.opt.z_dim, input.size(2), input.size(3)
                    )

                if label_to_shift is not None:
                    # recursive_check(z, 'tg: z')

                    z_layer = mask * z_layer + (1 - mask) * z
                    # recursive_check(z_layer, 'z_layer')

                seg = torch.cat((z_layer, input), dim=1)

            if i in shifted_noise_injection_layers:
                raise NotImplementedError("This is zombi code that may be revived")
                ##### shifted noise injection
                block_output = self.body[i](x, seg_shifted)
            else:
                ##### default
                block_output = self.body[i](x, seg)

            x = block_output["resnet_block_output"]

            # record outputs of all layers
            for key in block_output.keys():
                new_key = f"block_{i}_" + key
                intermediate_features_dict[new_key] = block_output[key]

            if i < self.opt.num_res_blocks - 1:
                x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        intermediate_features_dict["x"] = x

        if return_layers is None:
            return x
        else:
            output_dict = {}

            if "vgg_penultimate" in return_layers:
                intermediate_features_dict["vgg_penultimate"] = self.vgg(x)
            if "vgg_features" in return_layers:
                intermediate_features_dict["vgg_features"] = self.vgg(x)

            for features in return_layers:
                assert (
                    features in intermediate_features_dict.keys()
                ), f"Specify valid layers. The choices are {intermediate_features_dict.keys()}"
                output_dict[features] = intermediate_features_dict[features]

            if label_to_shift is not None:
                output_dict["mask"] = mask
            else:
                output_dict["mask"] = torch.zeros(
                    z.size(0), 3, seg.size(2), seg.size(3), device=z.device
                )

            return output_dict
