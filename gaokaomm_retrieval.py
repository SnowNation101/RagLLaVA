import clip
import torch
import numpy as np
from tqdm import tqdm
import faiss
from glob import glob
import os
import json
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_faiss_gaokaomm(val_dataset, device, model, preprocess=None):
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
            image = preprocess(Image.open(image_path)).to(device)
            image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))

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


def text_to_image(text, model, ind, topk=5):
    text_tokens = clip.tokenize([text], truncate=True)
    text_features = model.encode_text(text_tokens.to(device))

    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype("float32")

    D, I = ind.search(text_embeddings, topk)
    return D, I

def image_to_image(image, model, preprocess, ind, topk=5):
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_embeddings = image_features.cpu().detach().numpy().astype("float32")
    
    D, I = ind.search(image_embeddings, topk)
    
    return D, I


if __name__ == "__main__":

    val_dataset = []
    directory = 'datasets/GaokaoMM/'
    json_files = glob(os.path.join(directory, '*.json'))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file).get('example')
            val_dataset += data
    val_dataset = [item for item in val_dataset if item.get('year') != '2023']

    clip_model, preprocess = clip.load("ViT-L/14@336px", device="cuda", jit=False)

    index, index_to_image_id = build_faiss_gaokaomm(
        val_dataset,
        device,
        clip_model,
        preprocess=preprocess,
    )

    # faiss.write_index(
    #     index,
    #     "datasets/faiss_index/"
    #     + "GaokaoMM"
    #     + "_test_image"
    #     + ".index",
    # )

    # index = faiss.read_index(
    #     "datasets/faiss_index/"
    #     + 'GaokaoMM'
    #     + "_test_image"
    #     + ".index"
    # )

    # with open("datasets/GaokaoMM/2010-2023_Biology_MCQs.json", 'r', encoding='utf-8') as file:
    #     data = json.load(file)['example']
    #     questions = [item['question'] for item in data]

    target_set = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file).get('example')
            target_set += data
    target_set = [item for item in target_set if item.get('year') == '2023']

    questions = [item['question'] for item in target_set]
    for question in questions:
        D, I = text_to_image(question, clip_model, index, topk=5)

        print(f"Query: {question}")
        print("Top 5 similar images:")
        for i, idx in enumerate(I[0]):
            image_id = index_to_image_id[idx]
            print(f"  - Image ID: {image_id}, Distance: {D[0][i]}")

        print()

    # images = [item['picture'] for item in target_set]
    # for img_paths in images:
    #     for image in img_paths:
    #         img = Image.open('datasets/GaokaoMM/Data/'+image)
    #         D, I = image_to_image(img, clip_model, preprocess, index, topk=5)
            
    #         print(f"Image: {image}")
    #         print("Top 5 similar images:")
    #         for i, idx in enumerate(I[0]):
    #             image_id = index_to_image_id[idx]
    #             print(f"  - Image ID: {image_id}, Distance: {D[0][i]}")

    #         print()