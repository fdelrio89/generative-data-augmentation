#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from itertools import product

import pandas as pd
from PIL import Image
import ruamel.yaml as yaml
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('BLIP')

from models.blip import blip_decoder
from data import create_dataset, create_sampler, create_loader


class SimpleDataset:
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = transform(image)
        return image, self.image_paths[idx]

    def __len__(self):
        return len(self.image_paths)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# model_to_use = 'caption_base_flickr'
# model_to_use = 'caption_flickr_aae_color' # color
# model_to_use = 'caption_flickr_aae_counting' # counting
# model_to_use = 'caption_flickr_aae_gender' # gender
# model_to_use = 'caption_flickr_aae_color+counting+gender'
model_to_use = os.environ['MODEL_TO_USE']


base_ckpt_dir = 'BLIP/output'
model_dir = f'{base_ckpt_dir}/{model_to_use}'
ckpt_path = f'{model_dir}/checkpoint_best.pth'
config_path = f'{model_dir}/config.yaml'
feature_path = f'outputs/{model_to_use}_features_all.pth'


config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)


base_data_paths = {
    '/workspace1/fidelrio/flickr30k/imgs/': 'base',
    '/workspace1/fidelrio/sd_color_images/image_0_': 'generated_andres_color',
    '/workspace1/fidelrio/sd_counting_images/image_0_': 'generated_andres_counting',
    '/workspace1/fidelrio/sd_gender_images/image_0_': 'generated_andres_gender',
    '/workspace1/fidelrio/caspillaga/impainting/color/image_0_': 'generated_charlie_color',
    '/workspace1/fidelrio/caspillaga/impainting/counting/image_0_': 'generated_charlie_counting',
    '/workspace1/fidelrio/caspillaga/impainting/gender/image_0_': 'generated_charlie_gender',
    '/workspace1/fidelrio/aae/Caption_training_color/image_0_': 'generated_aae_color',
    '/workspace1/fidelrio/aae/Caption_training_counting/image_0_': 'generated_aae_counting',
    '/workspace1/fidelrio/aae/Caption_training_gender/image_0_': 'generated_aae_gender',
}


config['ann_root'] = '/mnt/nas2/GrimaRepo/fidelrio/generative-data-augmentation/data/'
config['image_root'] = {
    'all': '/workspace1/fidelrio/flickr30k/imgs/' ,
#     'train_color': '/workspace1/fidelrio/sd_color_images/image_0_',
#     'train_counting': '/workspace1/fidelrio/sd_counting_images/image_0_',
#     'train_gender': '/workspace1/fidelrio/sd_gender_images/image_0_',
}
config['image_exts'] = {
    'all': '.jpg',
#     'train_color': '.png',
#     'train_counting': '.png',
#     'train_gender': '.png',
}
config['list_skills'] = ''
config['available_skills'] = ''
# config['list_skills'] = 'color,counting,gender'
# config['available_skills'] = 'color,counting,gender'


dataset_name = config['dataset']
train_dataset, val_dataset, test_dataset = create_dataset(dataset_name, config)


skill_dfs = {
    (split, skill): f'Image_quality_{split}_{skill}.tsv' for split, skill in product(
        ['val', 'testing'], ['color', 'counting', 'gender']
    )
}

skill_dfs = {
    **skill_dfs,
    **{
    ('train', skill): f'data/Caption_training_{skill}.tsv' for skill in ['color', 'counting', 'gender']
}}

for (split, skill), df_path in skill_dfs.items():
    df = pd.read_csv(df_path, sep='\t')
    df['image_path'] = '/workspace1/fidelrio/flickr30k/imgs/' + df['image_ID'].astype(str) + '.jpg'
    skill_dfs[split, skill] = df

for skill in ['color', 'counting', 'gender']:
    skill_dfs['test', skill] = skill_dfs['testing', skill]



empty_skill_val = list(set(val_dataset.datasets[0].img_ids.keys())
                       - set(skill_dfs['val','color']['image_path'].tolist())
                       - set(skill_dfs['val','counting']['image_path'].tolist())
                       - set(skill_dfs['val','gender']['image_path'].tolist()))

empty_skill_test = list(set(test_dataset.datasets[0].img_ids.keys())
                       - set(skill_dfs['testing','color']['image_path'].tolist())
                       - set(skill_dfs['testing','counting']['image_path'].tolist())
                       - set(skill_dfs['testing','gender']['image_path'].tolist()))


color_sd_path = Path('/mnt/nas2/GrimaRepo/afcarvallo/sd_images_color_test/')
counting_sd_path = Path('/mnt/nas2/GrimaRepo/afcarvallo/sd_images_counting_test/')
gender_sd_path = Path('/mnt/nas2/GrimaRepo/afcarvallo/sd_images_gender_test/')

images_to_extract = {
    # 'base': list(test_dataset.datasets[0].img_ids.keys()),
    # 'none_base_val': empty_skill_val,
    # 'none_base_test': empty_skill_test,

    'color_base_train': skill_dfs['train','color']['image_path'].tolist(),
    'counting_base_train': skill_dfs['train','counting']['image_path'].tolist(),
    'gender_base_train': skill_dfs['train','gender']['image_path'].tolist(),

    'color_base_val': skill_dfs['val','color']['image_path'].tolist(),
    'counting_base_val': skill_dfs['val','counting']['image_path'].tolist(),
    'gender_base_val': skill_dfs['val','gender']['image_path'].tolist(),

    'color_base_test': skill_dfs['testing','color']['image_path'].tolist(),
    'counting_base_test': skill_dfs['testing','counting']['image_path'].tolist(),
    'gender_base_test': skill_dfs['testing','gender']['image_path'].tolist(),

    # 'color_sd': [str(p) for p in color_sd_path.glob('*.png')],
    # 'counting_sd': [str(p) for p in counting_sd_path.glob('*.png')],
    # 'gender_sd': [str(p) for p in gender_sd_path.glob('*.png')],
    **{
        f'neg_{skill}_base_{split}': list(
            set(split_base_list) - set(skill_dfs[split,skill]['image_path'].tolist())
        )
        for skill, (split, split_base_list) in
        product(['color', 'counting', 'gender'],
                [
                    ('test', test_dataset.datasets[0].img_ids.keys()),
                    ('val', val_dataset.datasets[0].img_ids.keys()),
                    ('train', train_dataset.datasets[0].img_ids.keys())
                ])
    }
}

# import json
# with open('outputs/probing_skill_splits.json', 'w') as fp:
#     json.dump(images_to_extract, fp)
# assert False

transform = test_dataset.datasets[0].transform
images_to_extract = {k: SimpleDataset(ip, transform) for k, ip in images_to_extract.items()}




model = blip_decoder(pretrained=ckpt_path,
                     image_size=config['image_size'],
                     vit=config['vit'],
                     vit_grad_ckpt=config['vit_grad_ckpt'],
                     vit_ckpt_layer=config['vit_ckpt_layer'],
                     prompt=config['prompt'],
                     med_config='BLIP/configs/med_config.json')
model = model.to(device)


only_keep_cls = False

model.eval()
try:
    extracted_features = torch.load(feature_path)
except FileNotFoundError:
    extracted_features = {}
with torch.no_grad():
    for image_category in images_to_extract:
        if image_category in extracted_features:
            continue

        print(f'Extracting Features of {image_category}')
        loader = DataLoader(images_to_extract[image_category],
                            batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        collected_features = []
        collected_paths = []
        for images, image_paths in tqdm(loader):
            images = images.to(device)
            features = model.visual_encoder(images)
            if only_keep_cls:
                features = features[:,0,:]

            collected_features.append(features.cpu())
            collected_paths.extend(image_paths)

        collected_features = torch.cat(collected_features)
        extracted_features[image_category] = {
            'features': collected_features,
            'image_paths': collected_paths,
        }

        torch.save(extracted_features, feature_path)


for image_category in extracted_features:
    print(image_category, extracted_features[image_category]['features'].shape)
