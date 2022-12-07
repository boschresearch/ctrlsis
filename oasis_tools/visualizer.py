
import os
import torch
import torch
from torchvision.utils import save_image
from torch.nn import functional as F

import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def _load_chardict():
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


def _write_on_image(text, image_tensor, font_pixel_height=20,
                    monochromatic=False, first_batch_element_only=False):
    char_dict_torch = _load_chardict()
    text_tensor = torch.cat([char_dict_torch[character]
                            for character in text], dim=2)
    ch, h, w = text_tensor.size()
    B, C, H, W = image_tensor.size()
    text_tensor = text_tensor.unsqueeze(0)
    new_height, new_width = font_pixel_height, int(font_pixel_height / h * w)
    text_tensor = F.interpolate(
        text_tensor,
        size=(
            new_height,
            new_width),
        mode='nearest')

    if monochromatic:
        text_tensor = text_tensor.sum(1, keepdim=True)
    image_tensor_with_text = image_tensor
    text_tensor = text_tensor[:, :, :H, :W]
    if first_batch_element_only:
        image_tensor_with_text[0, :, :new_height, :new_width] = text_tensor
    else:
        image_tensor_with_text[:, :, :new_height, :new_width] = text_tensor
    return image_tensor_with_text


def xorshift(seed):
    seed ^= seed << 13
    seed ^= seed >> 17
    seed ^= seed << 5
    seed %= int("ffffffff", 16)
    return seed


def get_fixed_colormap(num_classes):
    seed = 1234
    colormap = torch.zeros(num_classes * 3)
    for i in range(num_classes * 3):
        seed = xorshift(seed)
        colormap[i] = seed % 10000
    colormap = colormap.reshape(num_classes, 3) / 10000
    return colormap


def colorize_label(label, num_classes):
    label = label.long()
    if len(label.size()) == 4 and label.size(1) > 1:
        label = label.argmax(1)
    if len(label.size()) == 3:
        label = label.unsqueeze(1)
    B, _, H, W = label.size()
    label = label.view(B * H * W)
    colormap = get_fixed_colormap(num_classes)
    colormap = colormap.to(label.device)
    colormap[0, :] = 0.0
    colored_label = colormap[label]
    colored_label = colored_label.view(B, H, W, 3)
    colored_label = colored_label.permute(0, 3, 1, 2).contiguous()
    return colored_label


def pad_tensor(image_tensor, image_type, longest_sequence,
               longest_h, longest_w, pad_sequence=True):
    if image_type == 'label':
        image_tensor = colorize_label(image_tensor.contiguous(), 200)
    elif image_type == 'rgb':
        assert len(image_tensor.size()) == 4
        if image_tensor.size(1) == 2 or image_tensor.size(1) > 3:
            image_tensor = image_tensor.mean(dim=1, keepdim=True)
        if image_tensor.size(1) == 1:
            image_tensor = image_tensor.expand(-1, 3, -1, -1)
        image_tensor = (image_tensor - image_tensor.min()) / \
            (image_tensor.max() - image_tensor.min())

    if image_tensor.size(2) != longest_h or image_tensor.size(3) != longest_w:
        pad_image = 0.2 * torch.rand(size=(image_tensor.size(0),
                                     3, longest_h, longest_w), device=image_tensor.device)
        pad_image[:, :, :image_tensor.size(
            2), :image_tensor.size(3)] = image_tensor
        image_tensor = pad_image

    if pad_sequence and image_tensor.size(0) < longest_sequence:
        pad_image = 0.2 * torch.rand(
            size=(
                longest_sequence - image_tensor.size(0),
                3,
                longest_h,
                longest_w),
            device=image_tensor.device)
        image_tensor = torch.cat((image_tensor, pad_image), dim=0)

    return image_tensor


def pad_sequence(image_tensor, longest_sequence, longest_h, longest_w):
    if image_tensor.size(0) < longest_sequence:
        pad_image = 0.2 * torch.rand(
            size=(
                longest_sequence - image_tensor.size(0),
                3,
                longest_h,
                longest_w),
            device=image_tensor.device)
        image_tensor = torch.cat((image_tensor, pad_image), dim=0)
    return image_tensor


def save_visualizations(visuals_dict, save_folder, epoch,
                        iter, label_images=False, name='out'):
    longest_sequence = 0
    longest_h = 0
    longest_w = 0

    for value in visuals_dict.values():
        if 'list' in value[1]:
            longest_sequence = max(longest_sequence, len(value[0]))
            for img in value[0]:
                longest_h = max(longest_h, img.size(2))
                longest_w = max(longest_w, img.size(3))
        else:
            longest_sequence = max(longest_sequence, value[0].size(0))
            longest_h = max(longest_h, value[0].size(2))
            longest_w = max(longest_w, value[0].size(3))

    for key in visuals_dict.keys():
        image_tensor = visuals_dict[key][0]
        image_type = visuals_dict[key][1]
        if image_type == 'label':
            if len(image_tensor.size()) == 2:
                image_tensor = image_tensor.unsqueeze(1)
            if image_tensor.size(1) > 1:
                image_tensor = image_tensor.argmax(1, keepdim=True)
        if 'list' in image_type:
            img_list = []
            for img in image_tensor:
                if 'label' in image_type:
                    if len(img.size()) == 2:
                        img = img.unsqueeze(1)
                    if img.size(1) > 1:
                        img = img.argmax(1, keepdim=True)
                img = pad_tensor(
                    img,
                    image_type.replace(
                        '_list',
                        ''),
                    longest_sequence,
                    longest_h,
                    longest_w,
                    pad_sequence=False)
                img_list.append(img)
            image_tensor = torch.cat(img_list, 0)
            image_tensor = pad_sequence(
                image_tensor, longest_sequence, longest_h, longest_w)
        else:
            image_tensor = pad_tensor(
                image_tensor,
                image_type,
                longest_sequence,
                longest_h,
                longest_w)
        if label_images:
            image_tensor = _write_on_image(
                key,
                image_tensor,
                font_pixel_height=16,
                first_batch_element_only=True)
        visuals_dict[key] = image_tensor
    images = torch.cat([value for value in visuals_dict.values()])
    file_name = os.path.join(
        save_folder, f'{name}-ep-{epoch}-it-{iter}.jpg')
    save_image(images, file_name,
               nrow=longest_sequence, normalize=True)

    description = '\n'.join(
        [f'{i}: {key}' for i, key in enumerate(visuals_dict.keys())])
    keyhash = hash(description)
    file_name = os.path.join(
        save_folder, f'rows_{keyhash}.txt')
    with open(file_name, 'w') as file:
        file.write(description)
