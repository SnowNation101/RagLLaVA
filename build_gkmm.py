import faiss
import numpy as np
from PIL import Image
import clip
import torch
import json
import os
from glob import glob
from tqdm import tqdm
import argparse
import pandas as pd

from utils.model_series import load_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_faiss_gaokaomm(val_dataset, device, model, clip_type="clip", preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        pics = datum["picture"]

        for j in range(len(pics)):
            image_id = pics[j].split('/')[-1][10:-4]
            if image_id in index_to_image_id.values():
                continue
            image_path = "datasets/GaokaoMM/Data/" + pics[j]

        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(Image.open(image_path)).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(
                    pixel_values, mode=clip_type
                    ).to(torch.float)

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="GaokaoMM")
    parser.add_argument("--clip_type", type=str, default="clip")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_clip(args)

    if args.datasets == "GaokaoMM":
        val_dataset = []
        directory = 'datasets/GaokaoMM/'
        json_files = glob(os.path.join(directory, '*.json'))

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file).get('example')
                val_dataset += data
        val_dataset = [item for item in val_dataset if item.get('year') != '2023']

    index, index_to_image_id = build_faiss_gaokaomm(
        val_dataset,
        device,
        model,
        clip_type=args.clip_type,
        preprocess=preprocess,
    )

    faiss.write_index(
        index,
        "datasets/faiss_index/"
        + args.datasets
        + "_test_image"
        + ".index",
    )
