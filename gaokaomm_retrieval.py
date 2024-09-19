import clip
import torch
import numpy as np
from tqdm import tqdm
import faiss
from glob import glob
import os
import json
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))

def build_faiss_gaokaomm(val_dataset, device, model, preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        pics = datum['picture']

        for j in range(len(pics)):
            # image_id = pics[j].split('/')[-1][10:-4]
            image_id = pics[j]
            if image_id in index_to_image_id.values():
                continue
            image_path = 'datasets/GaokaoMM/Data/' + pics[j]

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

    embeddings = np.vstack(embeddings).astype('float32')

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id

def load_faiss_index(index_path, index_to_image_id_path):
    index = faiss.read_index(index_path)
    with open(index_to_image_id_path, 'r', encoding='utf-8') as f:
        index_to_image_id = json.load(f)
    return index, index_to_image_id


def text_to_image(text, model, ind, topk=5):
    text_tokens = clip.tokenize([text], truncate=True)
    text_features = model.encode_text(text_tokens.to(device))

    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype('float32')

    D, I = ind.search(text_embeddings, topk)
    return D, I

def image_to_image(image, model, preprocess, ind, topk=5):
    with torch.no_grad():
        img = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(img)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeddings = image_features.cpu().detach().numpy().astype('float32')
        
        D, I = ind.search(image_embeddings, topk)
    
    return D, I


if __name__ == '__main__':

    val_dataset = []
    directory = 'datasets/GaokaoMM/'
    json_files = glob(os.path.join(directory, '*.json'))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file).get('example')
            val_dataset += data
    val_dataset = [item for item in val_dataset if item.get('year') not in ['2023', '2022']]
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda', jit=False)

    index, index_to_image_id = build_faiss_gaokaomm(
        val_dataset,
        device,
        clip_model,
        preprocess=preprocess,
    )

    # faiss.write_index(
    #     index,
    #     'datasets/faiss_index/GaokaoMM_test_image.index'
    # )

    # with open('index_to_image_id.json', 'w', encoding='utf-8') as f:
    #     json.dump(index_to_image_id, f, ensure_ascii=False, indent=4)

    # index, index_to_image_id = load_faiss_index(
    #     'datasets/faiss_index/GaokaoMM_test_image.index',
    #     'index_to_image_id.json'
    # )

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file).get('example')
            data = [item for item in data if item.get('year') in ['2022', '2023']]

        output_data = {'results': []}

        for item in data:

            question = item['question']
            images = item['picture']

            result_entry = {
                'year': item['year'],
                'category': item['category'],
                'question': question,
                'picture': images,
                'answer': item['answer'],
                'analysis': item['analysis'],
                'index': item['index'],
                'score': item['score'],
                'retrieved_results': []
            }

            for image in images:
                img = Image.open('datasets/GaokaoMM/Data/'+image)
                D, I = image_to_image(img, clip_model, preprocess, index, topk=5)
                img_retrieve = {
                    'retrieved_img': [],
                    'distances': D[0].tolist()
                }
                img_result_per_image = []
                for idx, distance in zip(I[0], D[0]):
                    retrieved_img = index_to_image_id[idx]
                    img_result_per_image.append({
                        'retrieved_img': retrieved_img,
                        'distance': distance,
                    })
                result_entry['retrieved_results'].append(img_result_per_image)
                print('Image ' + image + ' Done!')

            output_data['results'].append(result_entry)
            

        # with open('image_retrieve.json', 'w', encoding='utf-8') as f:
        #     json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        def numpy_float32_default(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError

        with open('output/' + json_file.split('/')[-1], 'w', encoding='utf-8') as f:
            json.dump(output_data, f, default=numpy_float32_default, ensure_ascii=False, indent=4)



        # questions = [item['question'] for item in target_set]
        # # for question in questions:
        # #     D, I = text_to_image(question, clip_model, index, topk=5)

        # #     print(f'Query: {question}')
        # #     print('Top 5 similar images:')
        # #     for i, idx in enumerate(I[0]):
        # #         image_id = index_to_image_id[idx]
        # #         print(f'  - Image ID: {image_id}, Distance: {D[0][i]}')

        # #     print()

        # images = [item['picture'] for item in target_set]
        # for img_paths in images:
        #     for image in img_paths:
        #         img = Image.open('datasets/GaokaoMM/Data/'+image)
        #         D, I = image_to_image(img, clip_model, preprocess, index, topk=5)
                
        #         print(f'Image: {image}')
        #         print('Top 5 similar images:')
        #         for i, idx in enumerate(I[0]):
        #             image_id = index_to_image_id[idx]
        #             print(f'  - Image ID: {image_id}, Distance: {D[0][i]}')

        #         print()