import argparse
import os
from collections import defaultdict

import torch
from torch.nn import functional

from transformers import AutoFeatureExtractor

from PIL import Image

import pandas as pd

from tqdm import tqdm

from utils import load_config

parser = argparse.ArgumentParser(description="Image segmentation")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_name", type=str, default=None)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

model_name = config["pretrained_model_name"]

label2id = {k: v + 1 for k, v in config["label2id"].items()}
label2id["background"] = 0

id2label = {v: k for k, v in label2id.items()}

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
feature_extractor.reduce_labels = False

df = pd.read_csv(config["submission_frame_path"])

model = torch.load(args.model_path, map_location=device)
model.eval()

pred_classes = []
pred_segments = []
for file_name in tqdm(df["file_name"]):
    img = Image.open(os.path.join(config["test_image_directory"], file_name))
    encoded_inputs = feature_extractor(img, return_tensors="pt")
    pixel_values = encoded_inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        upsampled_logits = functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)[0]

    predicted = predicted.view(-1)

    labels = defaultdict(list)
    cnts = defaultdict(int)

    id = 0
    start = -1
    cnt = 0
    for idx, _id in enumerate(predicted.tolist()):
        if _id != id:
            if start == -1:
                start = idx
            else:
                labels[id].extend([start, cnt])
                cnts[id] += cnt

                if _id == 0:
                    start = -1
                else:
                    start = idx
                cnt = 0

            id = _id

        if _id == 0:
            continue

        cnt += 1

    max_cnt_id = sorted(cnts.keys(), key=lambda k: cnts[k])[-1]
    pred_classes.append(id2label[max_cnt_id])
    pred_segments.append(" ".join(map(str, labels[max_cnt_id])))

df["class"] = pred_classes
df["prediction"] = pred_segments

checkpoint = args.model_path.split("/")[-1]
train_id, trained_epochs = checkpoint.split(".")[0].split("_")

output_name = args.output_name
if output_name is None:
    output_name = f"{train_id}_{trained_epochs}_submission"
output_name = output_name.split(".")[0] + ".csv"

df.to_csv(output_name, index=False)
