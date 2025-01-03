#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import json
import os
from pathlib import Path
import re

import evaluate
from nltk.translate.meteor_score import meteor_score
import numpy as np
import pandas as pd
from tqdm.auto import trange, tqdm
import torch

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logging.getLogger("absl").setLevel(logging.CRITICAL)
logging.getLogger("evaluate").setLevel(logging.CRITICAL)


def get_image_id(composit_image_id):
    composit_image_id = composit_image_id.split('_')
    if len(composit_image_id) > 2:
        image_id = composit_image_id[-2]
    else:
        image_id = composit_image_id[-1]
    return image_id


def get_split(composit_image_id):
    composit_image_id = composit_image_id.split('_')
    if len(composit_image_id) > 2:
        composit_split = composit_image_id[:-2]
    else:
        composit_split = composit_image_id[:-1]

    split = '_'.join(composit_split)
    return split


def group_results(raw_results):
    results = {}
    for element in raw_results:
        image_id = get_image_id(element['image_id'])
        split = get_split(element['image_id'])
        results[image_id] = element['caption']
    return results


def pre_caption(caption, max_words=0):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if max_words and len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def load_testsets(path_text_data, list_skills=None):
    list_skills = list_skills if list_skills else []

    test_datasets = {'all' : pd.read_csv(path_text_data + 'Caption_all.tsv', sep='\t')}
    test_datasets.update({
        'test_%s' %skill : pd.read_csv(path_text_data + 'Caption_testing_%s.tsv'%skill, sep='\t') 
        for skill in list_skills if os.path.isfile(path_text_data + 'Caption_testing_%s.tsv'%skill)})
    test_datasets['test'] = test_datasets['all'][test_datasets['all'].split == 'test']
    test_datasets['val'] = test_datasets['all'][test_datasets['all'].split == 'val']
    del test_datasets['all']

    for split, dataset in test_datasets.items():
        dataset['caption'] = dataset['caption'].apply(pre_caption)
        dataset['image_id'] = dataset['image_ID'].astype(str)

    for split in test_datasets:
        test_datasets[split] = test_datasets[split][['image_id','caption']]

    return test_datasets


def to_grouped_dict(dataset):
    dict_dataset = dataset.to_dict(orient='records')
    grouped_dataset = defaultdict(list)
    for element in dict_dataset:
        grouped_dataset[element['image_id']].append(element['caption'])

    return grouped_dataset


def build_lists_for_evaluation(results, test_dataset, return_image_id=True):
    predictions = []
    references = []
    image_ids = list(test_dataset.keys())
    for image_id in image_ids:
        predictions.append(results[image_id])
        references.append(test_dataset[image_id])
    if return_image_id:
        return image_ids, predictions, references
    return predictions, references


path_text_data = './data/'
list_skills = ['color','counting','gender']
test_datasets = load_testsets(path_text_data, list_skills=list_skills)
test_datasets['test'].head()


for split in test_datasets:
    test_datasets[split] = to_grouped_dict(test_datasets[split])


cap_metrics = evaluate.combine(['bleu', 'rouge'])

def compute_metrics(predictions, references):
    metrics = cap_metrics.compute(predictions=predictions, references=references)
    for i in range(4):
        metrics[f'bleu{i+1}'] = metrics['precisions'][i]
    metrics['meteor'] = np.mean([meteor_score(hypothesis=p, references=rs) for p, rs in zip(predictions, references)])
    return metrics


base_dir = 'BLIP/output/'
exp_names = [
    str(dir_.stem) for dir_ in Path(base_dir).glob('*') if str(dir_.stem) not in ['saved_exps', '.gitignore']
]


best_epochs = {}
for exp_name in exp_names:
    ckpt_path = f'{base_dir}/{exp_name}/checkpoint_best.pth'
    if not os.path.exists(ckpt_path):
        continue
    ckpt = torch.load(ckpt_path)
    best_epochs[exp_name] = ckpt['epoch']


keep_exps = [
    'caption_flickr_aae_color',
    'caption_flickr_aae_counting',
    'caption_flickr_aae_gender',
    # 'caption_flickr_aae_color+counting+gender',
]
best_epochs = {k: best_epochs[k] for k in keep_exps}


# exp_name = exp_names[0]
# result_dir = Path(f'{base_dir}/{exp_name}/result')
# result_paths = list(result_dir.glob('test_epoch*.json'))
# result_path = result_paths[2]


# epoch = int(result_path.stem.replace('test_epoch', ''))
# with result_path.open() as fp:
#     raw_results = json.load(fp)
# results = group_results(raw_results)


output_dir = 'outputs/results'

print(f'Processing: {best_epochs}')

for exp_name, best_epoch in tqdm(best_epochs.items()):
    output_path = f'{output_dir}/{exp_name}.json'
    # if os.path.exists(output_path):
    #     continue

    print(f'Extracting results for {exp_name}')

    result_path = Path(f'{base_dir}/{exp_name}/result') / f'test_epoch{best_epoch}.json'
    print(f'Loading results in {result_path}')
    with result_path.open() as fp:
        raw_results = json.load(fp)
    results = group_results(raw_results)

    metrics = defaultdict(lambda: defaultdict(list))
    for test_name in tqdm(test_datasets):
        try:
            image_ids, predictions, references = build_lists_for_evaluation(
                results, test_datasets[test_name], return_image_id=True
            )
        except KeyError:
            continue

        print(f'Calculating {test_name} results')
        for image_id, prediction, reference in zip(image_ids, predictions, references):
            metrics[test_name][image_id].append({
                'prediction': prediction,
                'references': reference,
                'metrics': cap_metrics.compute(predictions=[prediction], references=[reference]),
            })

    with open(output_path, 'w') as fp:
        json.dump(metrics, fp)

    print(f'Results for {exp_name} saved in {output_path}')
