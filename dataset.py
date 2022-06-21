import os
from typing import Optional

from torch.utils.data import Dataset

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

from PIL import Image
from transformers import AutoFeatureExtractor


class HarborClassificationDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        img_names: np.ndarray,
        label_ids: np.ndarray,
        feature_extractor,
        transform=None
    ):
        if len(img_names) != len(label_ids):
            raise ValueError("img_names and label_ids must have the same length")

        self.img_dir = img_dir
        self.img_names = img_names
        self.label_ids = label_ids
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(self.img_dir, self.img_names[idx]))
        label_id = self.label_ids[idx]

        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]

        encoded_inputs = self.feature_extractor(img, return_tensors="pt")
        for k in encoded_inputs:
            encoded_inputs[k] = encoded_inputs[k].squeeze_()

        return encoded_inputs["pixel_values"], label_id

    def set_transform(self, transform):
        self.transform = transform

    def split(self, valid_fraction: float = 0.2, seed: Optional[int] = None):
        train_img_names, valid_img_names, train_label_ids, valid_label_ids = train_test_split(
            self.img_names, self.label_ids, test_size=valid_fraction, random_state=seed, stratify=self.label_ids
        )
        return self.__class__(
            self.img_dir,
            train_img_names,
            train_label_ids,
            self.feature_extractor,
            self.transform
        ), self.__class__(
            self.img_dir,
            valid_img_names,
            valid_label_ids,
            self.feature_extractor,
            self.transform
        )

    def stratified_k_fold(
        self,
        k: int = 5,
        shuffle: bool = False,
        seed: int = None,
    ):
        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for train_idxs, valid_idxs in skf.split(self.img_names, self.label_ids):
            yield self.__class__(
                self.img_dir,
                self.img_names[train_idxs],
                self.label_ids[train_idxs],
                self.feature_extractor,
                self.transform
            ), self.__class__(
                self.img_dir,
                self.img_names[valid_idxs],
                self.label_ids[valid_idxs],
                self.feature_extractor,
                self.transform
            )

    @classmethod
    def from_config(cls, config):
        data_config = config["images_without_mask"]
        data_dir, ext = data_config["directory"], data_config["extension"]

        label2id = config["label2id"]

        img_names = []
        label_ids = []
        for img_name in filter(lambda x: x.endswith(ext), os.listdir(data_dir)):
            img_names.append(img_name)
            label_ids.append(label2id["_".join(img_name.split("_")[:-1])])

        img_names = np.array(img_names)
        label_ids = np.array(label_ids)

        feature_extractor = AutoFeatureExtractor.from_pretrained(config["pretrained_model_name"])

        return cls(data_dir, img_names, label_ids, feature_extractor)


class HarborSegmentationDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_names: np.ndarray,
        mask_names: np.ndarray,
        label_ids: np.ndarray,
        feature_extractor,
        transform=None
    ):
        if len(img_names) != len(mask_names):
            raise ValueError("img_names and label_ids must have the same length")

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = img_names
        self.mask_names = mask_names
        self.label_ids = label_ids
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_names[idx]))
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[idx]))

        if self.transform is not None:
            transformed = self.transform(image=np.array(img), mask=np.array(mask))
            img = transformed["image"]
            mask = transformed["mask"]

        encoded_inputs = self.feature_extractor(img, mask, return_tensors="pt")
        for k in encoded_inputs:
            encoded_inputs[k] = encoded_inputs[k].squeeze_()

        return encoded_inputs

    def set_transform(self, transform):
        self.transform = transform

    def split(self, valid_fraction: float = 0.2, seed: Optional[int] = None):
        train_img_names, valid_img_names, train_mask_names, valid_mask_names, train_label_ids, valid_label_ids = train_test_split(
            self.img_names,
            self.mask_names,
            self.label_ids,
            test_size=valid_fraction,
            random_state=seed,
            stratify=self.label_ids
        )
        return self.__class__(
            self.img_dir,
            self.mask_dir,
            train_img_names,
            train_mask_names,
            train_label_ids,
            self.feature_extractor,
            self.transform
        ), self.__class__(
            self.img_dir,
            self.mask_dir,
            valid_img_names,
            valid_mask_names,
            valid_label_ids,
            self.feature_extractor,
            self.transform
        )

    def stratified_k_fold(
        self,
        k: int = 5,
        shuffle: bool = False,
        seed: int = None,
    ):
        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for train_idxs, valid_idxs in skf.split(self.img_names, self.label_ids):
            yield self.__class__(
                self.img_dir,
                self.mask_dir,
                self.img_names[train_idxs],
                self.mask_names[train_idxs],
                self.label_ids[train_idxs],
                self.feature_extractor,
                self.transform
            ), self.__class__(
                self.img_dir,
                self.mask_dir,
                self.img_names[valid_idxs],
                self.mask_names[valid_idxs],
                self.label_ids[valid_idxs],
                self.feature_extractor,
                self.transform
            )

    @classmethod
    def from_config(cls, config):
        data_config = config["images_with_mask"]
        img_dir, mask_dir = data_config["image_directory"], data_config["mask_directory"]
        img_ext, mask_ext = data_config["image_extension"], data_config["mask_extension"]

        label2id = config["label2id"]

        img_names = []
        label_ids = []
        for img_name in sorted(filter(lambda x: x.endswith(img_ext), os.listdir(img_dir))):
            img_names.append(img_name)
            label_ids.append(label2id["_".join(img_name.split("_")[:-1])])

        img_names = np.array(img_names)
        label_ids = np.array(label_ids)

        mask_names = [mask_name for mask_name in sorted(filter(lambda x: x.endswith(mask_ext), os.listdir(mask_dir)))]
        mask_names = np.array(mask_names)

        feature_extractor = AutoFeatureExtractor.from_pretrained(config["pretrained_model_name"])
        feature_extractor.reduce_labels = False

        return cls(img_dir, mask_dir, img_names, mask_names, label_ids, feature_extractor)
