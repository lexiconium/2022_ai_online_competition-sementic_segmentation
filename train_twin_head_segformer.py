import argparse
import os
import uuid

import numpy as np

import torch
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader

from datasets import load_metric

import albumentations
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from utils import set_seed, load_config
from dataset import HarborClassificationDataset, HarborSegmentationDataset
from model import TwinHeadSegformerForSemanticSegmentation

parser = argparse.ArgumentParser(description="Train twin head segformer")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--id", type=str, default=None)
parser.add_argument("--num_epochs", type=int, default=None)

args = parser.parse_args()

if args.seed is not None:
    set_seed(args.seed)

train_id = args.id
if train_id is None:
    train_id = uuid.uuid4().hex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
train_config = config["train"]["twin_head"]

model_name = config["pretrained_model_name"]

label2id = {k: v + 1 for k, v in config["label2id"].items()}
label2id["background"] = 0

id2label = {v: k for k, v in label2id.items()}

transform = albumentations.Compose([
    albumentations.CoarseDropout(
        max_holes=16, max_height=0.1, max_width=0.1, min_height=0.05, min_width=0.05, p=0.5
    ),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.SafeRotate(15, p=0.5),
    albumentations.GaussNoise(p=0.5),
    albumentations.OpticalDistortion(p=0.5),
    albumentations.OneOf([
        albumentations.RGBShift(),
        albumentations.RandomToneCurve(),
        albumentations.InvertImg(),
        albumentations.ToGray()
    ]),
    ToTensorV2()
])

classifier_train_dataset = HarborClassificationDataset.from_config(config)
classifier_train_dataset.set_transform(transform)

train_dataset = HarborSegmentationDataset.from_config(config)
train_dataset.set_transform(transform)

classifier_train_dataloader = DataLoader(
    classifier_train_dataset, batch_size=train_config["classifier_batch_size"], shuffle=True
)
train_dataloader = DataLoader(
    train_dataset, batch_size=train_config["batch_size"], shuffle=True
)

model = TwinHeadSegformerForSemanticSegmentation.from_pretrained(
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

accumulation_steps = train_config["accumulation_steps"]

losses = []
f1_metric = load_metric("f1")
miou_metric = load_metric("mean_iou")

model.train()

num_epochs = args.num_epochs
if num_epochs is None:
    num_epochs = train_config["num_epochs"]

step = 0
for epoch in range(1, num_epochs + 1):
    for (classifier_pixel_values, classifier_labels), segmenter_batch in tqdm(
        zip(classifier_train_dataloader, train_dataloader),
        train_id,
        total=min(len(classifier_train_dataloader), len(train_dataloader))
    ):
        step += 1

        classifier_pixel_values = classifier_pixel_values.to(device)
        classifier_labels = classifier_labels.to(device)
        pixel_values = segmenter_batch["pixel_values"].to(device)
        labels = segmenter_batch["labels"].to(device)

        outputs = model(
            classifier_pixel_values=classifier_pixel_values,
            classifier_labels=classifier_labels,
            pixel_values=pixel_values,
            labels=labels
        )
        classifier_logits, logits = outputs.classifier_logits, outputs.logits

        loss = (outputs.classifier_loss + outputs.loss) / 2
        loss /= accumulation_steps

        losses.append(loss.item())

        loss.backward()

        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        f1_metric.add_batch(
            predictions=classifier_logits.argmax(dim=-1).detach().cpu().numpy(),
            references=classifier_labels.detach().cpu().numpy()
        )

        if epoch % train_config["eval_frequency"] == 0:
            with torch.no_grad():
                upsampled_logits = functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)

                miou_metric.add_batch(
                    predictions=predicted.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy()
                )

    micro_f1 = f1_metric.compute(average="micro")["f1"]

    if epoch % train_config["eval_frequency"]:
        print(
            f"epoch: {epoch}\n"
            f"├─ loss: {np.mean(losses[-100:]):.6f}\n"
            f"└─ micro f1: {micro_f1:.4f}\n"
        )
    else:
        miou_metrics = miou_metric.compute(
            num_labels=len(id2label), ignore_index=label2id["background"], reduce_labels=False
        )

        print(
            f"epoch: {epoch}\n"
            f"├─ loss: {np.mean(losses[-100:]):.6f}\n"
            f"├─ micro f1: {micro_f1:.4f}\n"
            f"├─ mIoU: {miou_metrics['mean_iou']:.4f}\n"
            f"└─ mAcc: {miou_metrics['mean_accuracy']:.4f}\n"
        )

    torch.save(model, os.path.join(os.path.dirname(__file__), "checkpoints", f"{train_id}_{epoch}.pt"))
