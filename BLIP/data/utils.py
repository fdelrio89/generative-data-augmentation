import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=50):
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
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    )
    question = question.rstrip(' ')

    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)

    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result,open(final_result_file,'w'))
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}

    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval


# Generic Captioning Evaluation
import evaluate
from nltk.translate.meteor_score import meteor_score
import numpy as np
cap_metrics = evaluate.combine(['bleu', 'rouge'])

def group_results(raw_results):
    results = {}
    for element in raw_results:
        image_id = element['image_id']
        results[image_id] = element['caption']
    return results


def build_lists_for_evaluation(results, test_dataset):
    predictions = []
    references = []
    for image_id in test_dataset:
        if image_id not in results:
            print(f'Warning: {image_id} not present in prediction')
            continue
        predictions.append(results[image_id])
        references.append(test_dataset[image_id])
    return predictions, references


def generic_caption_eval(results, testset, split):
    results = group_results(results)
    predictions, references = build_lists_for_evaluation(results, testset)
    metrics = compute_metrics(predictions=predictions,
                              references=references)
    return metrics


def compute_metrics(predictions, references):
    metrics = cap_metrics.compute(predictions=predictions,
                                  references=references)
    for i in range(4):
        metrics[f'bleu{i+1}'] = metrics['precisions'][i]

    meteor_scores = [meteor_score(hypothesis=p, references=rs) for p, rs in zip(predictions, references)]
    metrics['meteor'] = np.mean(meteor_scores)
    return metrics

import re
from collections import defaultdict
import pandas as pd

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

    print("Text loaded for testing:", list(test_datasets.keys()))
    return test_datasets

def to_grouped_dict(dataset):
    dict_dataset = dataset.to_dict(orient='records')
    grouped_dataset = defaultdict(list)
    for element in dict_dataset:
        grouped_dataset[element['image_id']].append(element['caption'])

    return grouped_dataset

def load_test_val_data(path_text_data, list_skills):
    test_datasets = load_testsets(path_text_data, list_skills=list_skills)
    for split in test_datasets:
        test_datasets[split] = to_grouped_dict(test_datasets[split])
    return test_datasets
