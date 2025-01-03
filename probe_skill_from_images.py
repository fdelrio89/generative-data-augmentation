#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import copy
from itertools import product
import json
import os
import torch
import numpy as np
from PIL import Image
import random
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from tqdm.auto import tqdm
import ruamel.yaml as yaml
from sklearn.metrics import precision_recall_fscore_support

import sys
sys.path.append('BLIP')
from models.blip import blip_decoder
from data import create_dataset


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True


class SimpleDataset:
    def __init__(self, image_paths, targets, transform=None):
        assert len(image_paths) == len(targets), (f"image_path and targets must be same length "
                                                  f"{len(image_paths)} vs. {len(targets)}")

        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]

    def __len__(self):
        return len(self.image_paths)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Processing using {device}')


skill_to_use = os.environ['SKILL_TO_USE']
model_to_use = os.environ['MODEL_TO_USE']
# model_to_use = 'caption_base_flickr'
# model_to_use = 'caption_flickr_augmented_c' # color
# model_to_use = 'caption_flickr_augmented_counting' # counting
# model_to_use = 'caption_augmented_flickr' # gender
# model_to_use = 'caption_flickr_augmented_color+counting+gender'
print(f'Probing with {model_to_use} model')
print(f'Probing {skill_to_use} images')



base_ckpt_dir = 'BLIP/output'
model_dir = f'{base_ckpt_dir}/{model_to_use}'
ckpt_path = f'{model_dir}/checkpoint_best.pth'
config_path = f'{model_dir}/config.yaml'

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config['ann_root'] = '/mnt/ialabnas/homes/fidelrio/generative-data-augmentation/data/'
config['image_root'] = {'all': '/workspace1/fidelrio/flickr30k/imgs/' }
config['image_exts'] = {'all': '.jpg'}
config['list_skills'] = ''
config['available_skills'] = ''


dataset_name = config['dataset']
train_dataset, val_dataset, test_dataset = create_dataset(dataset_name, config)

print("Loading base model")
base_model = blip_decoder(
    pretrained=ckpt_path,
    image_size=config['image_size'],
    vit=config['vit'],
    vit_grad_ckpt=config['vit_grad_ckpt'],
    vit_ckpt_layer=config['vit_ckpt_layer'],
    prompt=config['prompt'],
    med_config='BLIP/configs/med_config.json'
)
base_model = base_model.visual_encoder
base_model = base_model.to(device)
for param in base_model.parameters():
    param.requires_grad = False
base_model.eval()
print("Base model loaded")


with open('outputs/probing_skill_splits.json') as fp:
    probing_dataset = json.load(fp)


# group = ['base', 'color_base_val', 'counting_base_val', 'gender_base_val']
# group = {'none_base_val': 0, 'color_base_val': 1, 'counting_base_val': 2, 'gender_base_val': 3,
#          'none_base_test': 0, 'color_base_test': 1, 'counting_base_test': 2, 'gender_base_test': 3}

if skill_to_use == 'color':
    group_name = 'color'
    group = {'color_base_val': 1, 'neg_color_base_val': 0,
             'color_base_test': 1, 'neg_color_base_test': 0,
             'color_base_train': 1, 'neg_color_base_train': 0,
            }
elif skill_to_use == 'counting':
    group_name = 'counting'
    group = {'counting_base_val': 1, 'neg_counting_base_val': 0,
             'counting_base_test': 1, 'neg_counting_base_test': 0,
             'counting_base_train': 1, 'neg_counting_base_train': 0,
            }
elif skill_to_use == 'gender':
    group_name = 'gender'
    group = {'gender_base_val': 1, 'neg_gender_base_val': 0,
             'gender_base_test': 1, 'neg_gender_base_test': 0,
             'gender_base_train': 1, 'neg_gender_base_train': 0,
            }


# samples = 800
# samples = 1.
# samples_by_cat = {
#     'color_base_train': 30526,
#     'neg_color_base_train': 13499,
#     'color_base_val': 862,
#     'neg_color_base_val': 568,
#     'color_base_test': 818,
#     'neg_color_base_test': 559,

#     'counting_base_train': 26392,
#     'neg_counting_base_train': 19128,
#     'counting_base_val': 973,
#     'neg_counting_base_val': 660,
#     'counting_base_test': 993,
#     'neg_counting_base_test': 635,

#     'gender_base_train': 75494,
#     'neg_gender_base_train': 5879,
#     'gender_base_val': 2570,
#     'neg_gender_base_val': 220,
#     'gender_base_test': 2641,
#     'neg_gender_base_test': 193,
# }

samples_by_cat = {
    'color_base_train': 5879,
    'neg_color_base_train': 5879,
    'color_base_val': 220,
    'neg_color_base_val': 220,
    'color_base_test': 193,
    'neg_color_base_test': 193,

    'counting_base_train': 5879,
    'neg_counting_base_train': 5879,
    'counting_base_val': 220,
    'neg_counting_base_val': 220,
    'counting_base_test': 193,
    'neg_counting_base_test': 193,

    'gender_base_train': 5879,
    'neg_gender_base_train': 5879,
    'gender_base_val': 220,
    'neg_gender_base_val': 220,
    'gender_base_test': 193,
    'neg_gender_base_test': 193,
}

for image_category in probing_dataset:
    if image_category not in group:
        continue
    samples = samples_by_cat[image_category]
    n = len(probing_dataset[image_category])
    if samples <= 1.:
        n_to_sample = int(samples * n)
    else:
        n_to_sample = min(samples, n)

    probing_dataset[image_category] = random.sample(probing_dataset[image_category], n_to_sample)


# p_train = 0.75
# p_val = 0.125
p_train = 0.8
p_val = 0.1

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []
for image_category in probing_dataset:
    if image_category not in group:
        continue

    n = len(probing_dataset[image_category])
    n_train = int(n * p_train)
    n_val = int(n * p_val)
    n_test = n - n_train - n_val

    x_train.extend(probing_dataset[image_category][:n_train])
    y_train.extend([group[image_category]]*n_train)

    x_val.extend(probing_dataset[image_category][n_train:n_train+n_val])
    y_val.extend([group[image_category]]*n_val)

    x_test.extend(probing_dataset[image_category][n_train+n_val:])
    y_test.extend([group[image_category]]*n_test)


transform = test_dataset.datasets[0].transform

train_dataset = SimpleDataset(x_train, y_train, transform=transform)
val_dataset = SimpleDataset(x_val, y_val, transform=transform)
test_dataset = SimpleDataset(x_test, y_test, transform=transform)

batch_size = 128
num_workers = 8
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=torch.cuda.is_available(),
                          num_workers=num_workers)
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=num_workers)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         pin_memory=torch.cuda.is_available(),
                         num_workers=num_workers)

print("Probing datasets created")


n_features = base_model(train_dataset[0][0].unsqueeze(0).to(device)).shape[-1]
def create_linear_probe(hidden_size):
    if isinstance(hidden_size, list):
        hidden_sizes = hidden_size
    else:
        hidden_sizes = [hidden_size]

    # n_features = x_train.shape[1]
    n_targets = len(set(group.values()))
    if n_targets == 2:
        n_targets = 1

    prev_hidden_size = hidden_sizes.pop(0)
    layers = [nn.Linear(n_features, prev_hidden_size)]
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_hidden_size, hidden_size))
        prev_hidden_size = hidden_size
    layers.append(nn.Linear(prev_hidden_size, n_targets))
    linear_probe = nn.Sequential(*layers)

    for p in linear_probe:
        try:
            nn.init.kaiming_uniform_(p.weight)
        except AttributeError:
            pass

    linear_probe = linear_probe.to(device)
    return linear_probe


def train(linear_probe,
          optimizer,
          train_loader,
          val_loader,
          num_epochs,
          patience):

    wait = 0
    best_model = None
    best_val_relevant_metric = 0

    is_binary_task = len(set(group.values())) == 2
    criterion = F.binary_cross_entropy_with_logits if is_binary_task else F.cross_entropy

    linear_probe.train()

    metrics = defaultdict(list)
    for epoch in tqdm(range(num_epochs)):
        linear_probe.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                x = base_model(x).mean(1)
            logit_pred = linear_probe(x).squeeze()
            y = y.float() if is_binary_task else y
            loss = criterion(logit_pred, y)
            loss.backward()
            optimizer.step()

            metrics['loss'].append(float(loss))
            if is_binary_task:
                pred = torch.sigmoid(logit_pred) > 0.5
                train_acc = (pred == y).sum() / y.shape[0]
                metrics['train_acc'].append(float(train_acc))
                prec, recall, f1 = calc_prec_recall_f1(pred, y)
                metrics['train_prec'].append(float(prec))
                metrics['train_recall'].append(float(recall))
                metrics['train_f1'].append(float(f1))
            else:
                pred = logit_pred.argmax(-1)
                train_acc = (pred == y).sum() / y.shape[0]
                metrics['train_acc'].append(float(train_acc))


        linear_probe.eval()
        cum_sum = 0
        cum_loss = 0
        cum_logit_pred, cum_y = [], []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            bsz = x.shape[0]
            with torch.no_grad():
                x = base_model(x).mean(1)
            logit_pred = linear_probe(x).squeeze()
            y = y.float() if is_binary_task else y
            loss = criterion(logit_pred, y)
            cum_loss += float(loss) * bsz

            cum_logit_pred.append(logit_pred)
            cum_y.append(y)

        logit_pred = torch.cat(cum_logit_pred)
        y = torch.cat(cum_y)

        val_loss = cum_loss / len(val_dataset)
        metrics['val_loss'].append(float(val_loss))
        if is_binary_task:
            pred = torch.sigmoid(logit_pred) > 0.5
            val_acc = (pred == y).sum() / y.shape[0]
            metrics['val_acc'].append(float(val_acc))
            prec, recall, f1 = calc_prec_recall_f1(pred, y)
            metrics['val_prec'].append(float(prec))
            metrics['val_recall'].append(float(recall))
            metrics['val_f1'].append(float(f1))
        else:
            pred = logit_pred.argmax(-1)
            val_acc = (pred == y).sum() / y.shape[0]
            metrics['val_acc'].append(float(val_acc))

        # Early stopping
        val_relevant_metric = metrics['val_f1'][-1] if is_binary_task else metrics['val_acc'][-1]
        if val_relevant_metric > best_val_relevant_metric:
            best_val_relevant_metric = val_relevant_metric
            best_model = copy.deepcopy(linear_probe)
            wait = 0
        else:
            wait += 1

        if wait > patience:
            break

    print(f'Finished in epoch {epoch}')
    return best_model, metrics, best_val_relevant_metric

def calc_prec_recall_f1(pred, y):
    p, r, f1, _ = precision_recall_fscore_support(
        y.cpu().detach().numpy(),
        pred.cpu().detach().numpy(),
        beta=1,
    )
    return p[1], r[1], f1[1]


is_binary_task = len(set(group.values())) == 2
relevant_metric_name = 'f1' if is_binary_task else 'acc'

num_epochs = 300
# num_epochs = 1
patience = 10

all_metrics = {}
grid_best_relevant_metric = 0
grid_best_model = None
grid_best_metrics = None
best_hidden_size = None
best_lr = None
print('Starting probing training grid search...')
# for exp_idx, (hidden_size, lr) in enumerate(product([64],
#                                                     [1e-2]), start=1):
for exp_idx, (hidden_size, lr) in enumerate(product([64, 128, 256],
                                                    [1e-2]), start=1):

    print(f'Start experiment: {exp_idx} with hidden_size={hidden_size} & lr={lr}')

    linear_probe = create_linear_probe(hidden_size=hidden_size)
    optimizer = SGD(linear_probe.parameters(), lr=lr, momentum=0.9)

    best_model, metrics, best_relevant_metric = train(linear_probe,
                                                      optimizer,
                                                      train_loader,
                                                      val_loader,
                                                      num_epochs=num_epochs,
                                                      patience=patience,
                                                     )

    all_metrics[(hidden_size, lr)] = metrics
    if  best_relevant_metric > grid_best_relevant_metric:
        best_hidden_size = hidden_size
        best_lr = lr
        grid_best_model = best_model
        grid_best_metrics = metrics
        grid_best_relevant_metric = best_relevant_metric


    linear_probe = optimizer = None

    print(f'hidden_size={hidden_size} lr={lr}')
    print(f'Best val {relevant_metric_name}: ', float(best_relevant_metric))
    print('\n')

print('best_hidden_size:', best_hidden_size)
print('best_lr:', best_lr)


is_binary_task = len(set(group.values())) == 2
criterion = F.binary_cross_entropy_with_logits if is_binary_task else F.cross_entropy

print("Starting Training Evaluation...")
grid_best_model.eval()
cum_loss = 0
cum_logit_pred = []
cum_y = []
for x, y in tqdm(test_loader):
    x, y = x.to(device), y.to(device)
    bsz = x.shape[0]
    with torch.no_grad():
        x = base_model(x).mean(1)
    logit_pred = grid_best_model(x).squeeze()
    y = y.float() if is_binary_task else y
    cum_logit_pred.append(logit_pred)
    cum_y.append(y)
    loss = criterion(logit_pred, y)
    cum_loss += float(loss) * bsz


logit_pred = torch.cat(cum_logit_pred)
y = torch.cat(cum_y)
grid_best_metrics['test_loss'] = (cum_loss / len(test_dataset))
if is_binary_task:
    pred = torch.sigmoid(logit_pred) > 0.5
    train_acc = (pred == y).sum() / y.shape[0]
    grid_best_metrics['test_acc'] = float(train_acc)
    prec, recall, f1 = calc_prec_recall_f1(pred, y)
    grid_best_metrics['test_prec'] = float(prec)
    grid_best_metrics['test_recall'] = float(recall)
    grid_best_metrics['test_f1'] = float(f1)
else:
    pred = logit_pred.argmax(-1)
    test_acc = (pred == y).sum() / y.shape[0]
    grid_best_metrics['test_acc'] = float(test_acc)


print('test_acc: ', grid_best_metrics['test_acc'])
if is_binary_task:
    print('test_prec', grid_best_metrics['test_prec'])
    print('test_recall', grid_best_metrics['test_recall'])
    print('test_f1', grid_best_metrics['test_f1'])


exps_to_run = 1

track_metric = 'test_f1' if is_binary_task else 'test_acc'
best_test_metrics = defaultdict(list)
print("Starting Test Evaluation...")
for _ in range(exps_to_run):
    test_metrics = {}
    hidden_size, lr = best_hidden_size, best_lr
    linear_probe = create_linear_probe(hidden_size=hidden_size)
    optimizer = SGD(linear_probe.parameters(), lr=lr, momentum=0.9)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    best_model, metrics, best_relevant_metric = train(linear_probe,
                                                             optimizer,
                                                             train_loader,
                                                             val_loader,
                                                             num_epochs=num_epochs,
                                                             patience=patience,
                                                            )


    is_binary_task = len(set(group.values())) == 2
    criterion = F.binary_cross_entropy_with_logits if is_binary_task else F.cross_entropy

    grid_best_model.eval()
    cum_loss = 0
    cum_logit_pred = []
    cum_y = []
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        bsz = x.shape[0]
        with torch.no_grad():
            x = base_model(x).mean(1)
        logit_pred = best_model(x).squeeze()
        y = y.float() if is_binary_task else y
        cum_logit_pred.append(logit_pred)
        cum_y.append(y)
        loss = criterion(logit_pred, y)
        cum_loss += float(loss) * bsz


    logit_pred = torch.cat(cum_logit_pred)
    y = torch.cat(cum_y)
    test_metrics['test_loss'] = (cum_loss / len(test_dataset))
    if is_binary_task:
        pred = torch.sigmoid(logit_pred) > 0.5
        train_acc = (pred == y).sum() / y.shape[0]
        test_metrics['test_acc'] = float(train_acc)
        prec, recall, f1 = calc_prec_recall_f1(pred, y)
        test_metrics['test_prec'] = float(prec)
        test_metrics['test_recall'] = float(recall)
        test_metrics['test_f1'] = float(f1)
    else:
        pred = logit_pred.argmax(-1)
        test_acc = (pred == y).sum() / y.shape[0]
        test_metrics['test_acc'] = float(test_acc)

    for metric_name in test_metrics:
        best_test_metrics[metric_name].append(test_metrics[metric_name])

    linear_probe = optimizer = best_model = None


print(model_to_use)
print(group_name)
for metric_name in best_test_metrics:
    print(f'mean {metric_name}: {np.mean(best_test_metrics[metric_name]):.4f}')
    print(f'std {metric_name}: {np.std(best_test_metrics[metric_name]):.4f}')
    print(best_test_metrics[metric_name])


result_path = 'outputs/results/skill-probe-all-feats/results.pth'
try:
    all_results = torch.load(result_path)
except FileNotFoundError:
    all_results = []

if 'aae' in model_to_use:
    model_to_use = model_to_use + '_v2'

all_results.append(
    {
        'model': model_to_use,
        'skill': group_name,
        'hidden_size': best_hidden_size,
        'lr': lr,
        'test_metrics': best_test_metrics
    }
)
torch.save(all_results, result_path)
