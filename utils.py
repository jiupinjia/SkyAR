import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim as sk_cpt_ssim

import os
import glob
import random

import torch

if torch.cuda.is_available():
    torch.cuda.current_device()

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils

import json


class PairedDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_random_brightness=False,
            with_random_gamma=False,
            with_random_saturation=False
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_random_brightness = with_random_brightness
        self.with_random_gamma = with_random_gamma
        self.with_random_saturation = with_random_saturation

    def transform(self, img1, img2):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size], interpolation=3)
        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size], interpolation=3)

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=(0.5, 1.0), ratio=(0.9, 1.1))
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))

        if self.with_random_brightness and random.random() > 0.5:
            # multiply a random number within a - b
            img1 = TF.adjust_brightness(img1, brightness_factor=random.uniform(0.5, 1.5))

        if self.with_random_gamma and random.random() > 0.5:
            # img**gamma
            img1 = TF.adjust_gamma(img1, gamma=random.uniform(0.5, 1.5))

        if self.with_random_saturation and random.random() > 0.5:
            # saturation_factor, 0: grayscale image, 1: unchanged, 2: increae saturation by 2
            img1 = TF.adjust_saturation(img1, saturation_factor=random.uniform(0.5, 1.5))

        # to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        return img1, img2


class CVPR2020_ADE20K_DEGF_Dataset(Dataset):

    def __init__(self, root_dir, img_size, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        if is_train:
            self.img_dirs = glob.glob(os.path.join(self.root_dir, 'images/train', '*.jpg'))
            self.augm = PairedDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True,
                with_random_brightness=True,
                with_random_gamma=True,
                with_random_saturation=True
            )
        else:
            self.img_dirs = glob.glob(os.path.join(self.root_dir, 'images/val', '*.jpg'))
            self.augm = PairedDataAugmentation(
                img_size=self.img_size
            )

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_A = cv2.imread(self.img_dirs[idx], cv2.IMREAD_COLOR)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

        p = self.img_dirs[idx].replace('images', 'density_estimation+guided_filter').replace('.jpg', '.png')
        img_B = cv2.imread(p, cv2.IMREAD_COLOR)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        img_A, img_B = self.augm.transform(img_A, img_B)

        data = {'A': img_A, 'B': img_B}

        return data


class CVPR2020_ADE20K_GF_Dataset(Dataset):

    def __init__(self, root_dir, img_size, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        if is_train:
            self.img_dirs = glob.glob(os.path.join(self.root_dir, 'images/train', '*.jpg'))
            self.augm = PairedDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True,
                with_random_brightness=True,
                with_random_gamma=True,
                with_random_saturation=True
            )
        else:
            self.img_dirs = glob.glob(os.path.join(self.root_dir, 'images/val', '*.jpg'))
            self.augm = PairedDataAugmentation(
                img_size=self.img_size
            )

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_A = cv2.imread(self.img_dirs[idx], cv2.IMREAD_COLOR)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

        p = self.img_dirs[idx].replace('images', 'guided_filter').replace('.jpg', '.png')
        img_B = cv2.imread(p, cv2.IMREAD_COLOR)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        img_A, img_B = self.augm.transform(img_A, img_B)

        data = {'A': img_A, 'B': img_B}

        return data


def get_loaders(args):

    if args.dataset == 'cvprw2020-ade20K-defg':
        training_set = CVPR2020_ADE20K_DEGF_Dataset(
            root_dir=r'./datasets/cvprw2020_sky_seg', img_size=args.in_size, is_train=True)
        val_set = CVPR2020_ADE20K_DEGF_Dataset(
            root_dir=r'./datasets/cvprw2020_sky_seg', img_size=args.in_size, is_train=False)
    elif args.dataset == 'cvprw2020-ade20K-fg':
        training_set = CVPR2020_ADE20K_GF_Dataset(
            root_dir=r'./datasets/cvprw2020_sky_seg', img_size=args.in_size, is_train=True)
        val_set = CVPR2020_ADE20K_GF_Dataset(
            root_dir=r'./datasets/cvprw2020_sky_seg', img_size=args.in_size, is_train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [maps, flowers, facades])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def parse_config(path_to_json=r'./config.json'):

    with open(path_to_json) as f:
      data = json.load(f)
    args = Struct(**data)

    return args


def clip_01(x):
    x[x>1.0] = 1.0
    x[x<0] = 0
    return x


def cpt_pxl_cls_acc(pred_idx, target):
    pred_idx = torch.reshape(pred_idx, [-1])
    target = torch.reshape(target, [-1])
    return torch.mean((pred_idx.int()==target.int()).float())


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2, dim=[1,2,3])
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return torch.mean(psnr)


def cpt_psnr(img, img_gt, PIXEL_MAX):
    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


def cpt_rgb_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    SSIM = 0
    for i in range(3):
        tmp = img[:, :, i]
        tmp_gt = img_gt[:, :, i]
        ssim = sk_cpt_ssim(tmp, tmp_gt)
        SSIM = SSIM + ssim
    return SSIM / 3.0


def cpt_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    return sk_cpt_ssim(img, img_gt)

