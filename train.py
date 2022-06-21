import argparse
import os
import uuid

import numpy as np

import torch
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader

from transformers import AutoModelForSemanticSegmentation
from datasets import load_metric

import albumentations
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from utils import set_seed, load_config
from dataset import HarborSegmentationDataset

parser = argparse.ArgumentParser(description="Train segmenter")
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()

if args.seed is not None:
    set_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_id = uuid.uuid4().hex[:16]

config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
train_config = config["train"]["segmenter"]

model_name = config["pretrained_model_name"]

label2id = {k: v + 1 for k, v in config["label2id"].items()}
label2id["background"] = 0

id2label = {v: k for k, v in label2id.items()}

transform = albumentations.Compose([
    albumentations.CoarseDropout(
        max_holes=16, max_height=0.1, max_width=0.1, min_height=0.05, min_width=0.05, p=0.5
    ),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.OneOf([
        albumentations.GaussNoise(p=0.5),
        albumentations.OpticalDistortion(p=0.5),
        albumentations.ToGray()
    ]),
    ToTensorV2()
])

dataset = HarborSegmentationDataset.from_config(config)
dataset.set_transform(transform)

dataloader = DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=True)

model = AutoModelForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=float(train_config["learning_rate"]),
    weight_decay=float(train_config["weight_decay"])
)

losses = []
metric = load_metric("mean_iou")

model.train()

for epoch in range(1, train_config["num_epochs"] + 1):
    for batch in tqdm(dataloader, train_id):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        with torch.no_grad():
            upsampled_logits = functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            metric.add_batch(
                predictions=predicted.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )

    metrics = metric.compute(
        num_labels=len(id2label), ignore_index=label2id["background"], reduce_labels=False
    )

    print(
        f"epoch: {epoch}\n"
        f"├─ loss: {np.mean(losses[-100:]):.6f}\n"
        f"├─ mIoU: {metrics['mean_iou']:.4f}\n"
        f"└─ mAcc: {metrics['mean_accuracy']:.4f}\n"
    )

    torch.save(
        model,
        os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
            f"{train_id}_{model_name.split('/')[-1]}_segmenter_epoch_{epoch}.pt"
        )
    )
