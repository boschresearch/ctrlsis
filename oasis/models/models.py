from oasis.models.sync_batchnorm import DataParallelWithCallback
import oasis.models.generator as generators
from oasis.models import transparent_generator
import oasis.models.discriminator as discriminators
import os
import copy
import torch
import torch.nn as nn

from enum import Enum
from torch.nn import init
import torch.nn.functional as F
import oasis.models.losses as losses


class Label(Enum):
    FAKE = 0
    REAL = 1


class OASIS_model(nn.Module):
    def __init__(self, opt, wrapped=False):
        super(OASIS_model, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if wrapped:
            print("Loading wrapped generator")
            self.netG = transparent_generator.OASIS_Generator(opt)
        else:
            print("Loading standard generator")
            self.netG = generators.OASIS_Generator(opt)

        # if opt.phase == "train":
        self.netD = discriminators.OASIS_Discriminator(opt)

        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None

        self.print_parameter_count()
        self.init_networks()

        # --- load previous checkpoints if needed ---
        self.load_checkpoints()

        # --- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer, noise_tensor=None):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0

            fake = self.netG(label)

            loss_G_adv = 0.0
            if self.opt.lambda_bneck > 0.0:
                output_D, bneck = self.netD(fake, return_bottleneck=True)
                loss_G_adv += F.binary_cross_entropy_with_logits(
                    bneck, torch.full_like(bneck, Label.REAL.value)
                )
            else:
                output_D = self.netD(fake)

            loss_G_adv += losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv

            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            loss_D = 0

            with torch.no_grad():
                fake = self.netG(label)

            loss_D_fake = 0.0
            if self.opt.lambda_bneck > 0.0:
                output_D_fake, bneck = self.netD(fake, return_bottleneck=True)
                loss_D_fake += F.binary_cross_entropy_with_logits(
                    bneck, torch.full_like(bneck, Label.FAKE.value)
                )
            else:
                output_D_fake = self.netD(fake)

            loss_D_fake += losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake

            loss_D_real = 0.0
            if self.opt.lambda_bneck > 0.0:
                output_D_real, bneck = self.netD(image, return_bottleneck=True)
                loss_D_real += F.binary_cross_entropy_with_logits(
                    bneck, torch.full_like(bneck, Label.REAL.value)
                )
            else:
                output_D_real = self.netD(image)

            loss_D_real += losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real

            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(
                    mask, output_D_mixed, output_D_fake, output_D_real
                )
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label)
                else:
                    fake = self.netEMA(label, noise_tensor)
            return fake

        if mode == "segment":
            with torch.no_grad():
                pred = self.netD(image)

            return pred

    def load_checkpoints(self):
        print("loading checkpoints")
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(
                self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_"
            )

            if hasattr(self, "netD"):
                # self.netD.load_state_dict(torch.load(path + "D.pth"))  # TODO: Change Back
                pass

            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
                print(path + "G.pth")
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
                print(path + "EMA.pth")

            print("loaded ckpt!")
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(
                self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_"
            )

            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
            print("loaded ckpt!")
        else:
            print("loaded nothing!")

    def print_parameter_count(self):
        networks = []

        if hasattr(self, "netG"):
            networks.append(self.netG)
        if hasattr(self, "netEMA"):
            networks.append(self.netEMA)
        if hasattr(self, "netD"):
            networks.append(self.netD)

        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (
                    isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)
                ):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print(
                "Created",
                network.__class__.__name__,
                "with %d parameters" % param_count,
            )

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


 


 

def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to(
            device=target_map.device
        )
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map
