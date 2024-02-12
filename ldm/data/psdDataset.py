import os

import PIL
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# if __name__ == '__main__':
#     with open(r'\Users\Jonathan\PycharmProjects\latent-diffusion\data\psd\psd_validation.txt', 'r',  encoding="utf8") as file:
#         for line in file:
#             #print(line)
#             image_lbl_pairs = [line.split(maxsplit=1)]
#             relative_path = [s[0] for s in image_lbl_pairs][0]
#             absolute_path = [os.path.join(r'\Users\Jonathan\PycharmProjects\latent-diffusion\data/psd/images', relative_path)][0]
#             print(relative_path)
#             print(absolute_path)
#             image = Image.open(absolute_path)
#             img = np.array(image).astype(np.uint8)


class PSDDataset(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 config=None,
                 size=None,
                 random_crop=True,
                 interpolation="bicubic",
                 flip_p=0.5):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.txt_file = txt_file
        self.data_root = data_root
        self._load()
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        # self.random_crop = random_crop
        # if self.size is not None and self.size > 0:
        #     self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        #     if not self.random_crop:
        #         self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        #     else:
        #         self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
        #     self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        # else:
        #     self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.absolute_paths)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["absolute_path"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example

    # def preprocess_image(self, absolute_path):
    #     image = Image.open(absolute_path)
    #     if not image.mode == "RGB":
    #         image = image.convert("RGB")
    #     image = np.array(image).astype(np.uint8)
    #     image = self.preprocessor(image=image)["image"]
    #     image = (image / 127.5 - 1.0).astype(np.float32)
    #     return image

    def _load(self):
        with open(self.txt_file, "r") as f:
            self.image_lbl_pairs = [line.split(maxsplit=1) for line in f.read().splitlines()]
            self.relative_paths = [s[0] for s in self.image_lbl_pairs]
            self.psd_labels = torch.tensor([[[int(value) for value in s[1].split(",")]] for s in self.image_lbl_pairs])
            self.absolute_paths = [os.path.join(self.data_root, p) for p in self.relative_paths]

            self.labels = {
                "absolute_path": np.array(self.absolute_paths),
                "relative_path": np.array(self.relative_paths),
                "psd_label": self.psd_labels
            }


class PSDDatasetTrain(PSDDataset):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/psd/psd_train.txt", data_root="data/psd/images", **kwargs)


class PSDDatasetValidation(PSDDataset):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/psd/psd_validation.txt", data_root="data/psd/images",
                         flip_p=flip_p, **kwargs)
