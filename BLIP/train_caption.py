'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import comet_ml
from comet_ml import ExistingExperiment, Experiment
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule, EarlyStopper
from data import create_dataset, create_sampler, create_loader

from data.utils import save_result, coco_caption_eval, generic_caption_eval, load_test_val_data


def setup_comet(cfg, resume_experiment_key='', **kwargs):
    api_key = os.environ.get('COMET_KEY', '')
    workspace = os.environ.get('COMET_WORKSPACE', '')
    enabled = bool(api_key) and bool(workspace)
    disabled = not enabled

    if resume_experiment_key:
        experiment = ExistingExperiment(api_key=api_key, previous_experiment=resume_experiment_key)
        return experiment

    experiment = Experiment(api_key=api_key, parse_args=True, project_name='BLIP',
                            workspace=workspace, disabled=disabled)

    experiment_name = os.environ.get('EXP_NAME')
    if experiment_name:
        experiment.set_name(experiment_name)

    experiment.log_parameters(cfg)
    return experiment


def train(model, data_loader, optimizer, epoch, device, experiment=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50
    its_per_epoch = len(data_loader)

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)

        loss = model(image, caption)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if experiment:
            its = (its_per_epoch * epoch) + i
            metrics = {'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]}
            experiment.log_metrics(metrics, prefix='train', step=its, epoch=epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)

        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            img_id = img_id if isinstance(img_id, str) else img_id.item()
            result.append({"image_id": img_id, "caption": caption})

    return result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating captioning dataset")
    dataset_name = config['dataset']
    train_dataset, val_dataset, test_dataset = create_dataset(dataset_name, config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])

    #### Model ####
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                           prompt=config['prompt'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    early_stopper = EarlyStopper(patience=config['patience'])
    early_stopping_signal = torch.zeros(1).to(device)

    experiment = config['experiment']
    del config['experiment']

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, experiment=experiment)

        val_result = evaluate(model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')

        test_result = evaluate(model_without_ddp, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')

        if utils.is_main_process():
            if 'coco' in dataset_name:
                coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
                coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test')

                if args.evaluate:
                    log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                                 **{f'test_{k}': v for k, v in coco_test.eval.items()},
                                }
                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }

                    if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                        best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                        best_epoch = epoch
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in coco_val.eval.items()},
                                **{f'test_{k}': v for k, v in coco_test.eval.items()},
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                # torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint.{epoch}.pth'))

                test_datasets = load_test_val_data(config['ann_root'], list_skills=config['available_skills'].split(','))
                eval_metrics = {}
                for split in test_datasets:
                    results_to_use = val_result if split == 'val' else test_result
                    eval_metrics[split] = generic_caption_eval(results_to_use, test_datasets[split], split)
                    metrics_to_log = {}
                    for relevant_metric in ['bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'meteor']:
                        metrics_to_log[relevant_metric] = eval_metrics[split][relevant_metric]
                    experiment.log_metrics(metrics_to_log, prefix=split, epoch=epoch)

                eval_val, eval_test = eval_metrics['val'], eval_metrics['val']
                if eval_val['bleu4'] > best:
                    best = eval_val['bleu4']
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in eval_val.items()},
                             **{f'test_{k}': v for k, v in eval_test.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if early_stopper.early_stop(eval_val[config['early_stopping_metric']]):
                    early_stopping_signal += 1

        if args.evaluate:
            break

        dist.all_reduce(early_stopping_signal, op=dist.ReduceOp.SUM)
        dist.barrier()
        if early_stopping_signal >= 1:
            print("Finish training early due to early stopping")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    experiment = setup_comet(config)
    config['experiment'] = experiment

    main(args, config)