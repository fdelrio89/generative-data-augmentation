#!/usr/bin/env python
# coding: utf-8

# In[4]:

import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True


# In[5]:


base_ckpt_dir = 'BLIP/output'


model_to_use = os.environ['MODEL_TO_USE']
# model_to_use = 'caption_base_flickr'
# model_to_use = 'caption_flickr_augmented_c' # color
# model_to_use = 'caption_flickr_augmented_counting' # counting
# model_to_use = 'caption_augmented_flickr' # gender
# model_to_use = 'caption_flickr_augmented_color+counting+gender'
print(f'Probing with {model_to_use} model')

feature_path = f'outputs/{model_to_use}_features.pth'


# In[3]:


all_features = torch.load(feature_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[4]:


for image_category in all_features:
    print(image_category, len(all_features[image_category]['features']))


# In[5]:

skill_to_use = os.environ['SKILL_TO_USE']
print(f'Probing {skill_to_use} images')

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
for image_category in all_features:
    if image_category not in group:
        continue

    all_features[image_category]['original_features'] = all_features[image_category]['features']
    if all_features[image_category]['features'].ndim == 3:
        all_features[image_category]['features'] = all_features[image_category]['features'].flatten(0,1)

    n = all_features[image_category]['features'].shape[0]
    all_features[image_category]['group'] = torch.ones((n,), dtype=torch.long) * group[image_category]


# In[6]:


import random
# samples = 800
samples = 1.

for image_category in all_features:
    if image_category not in group:
        continue
    n = all_features[image_category]['features'].shape[0]
    if samples <= 1.:
        n_to_sample = int(samples * n)
    else:
        n_to_sample = samples
    indices = random.sample(list(range(n)), k=n_to_sample)
    indices = torch.tensor(indices)

    all_features[image_category]['features'] = all_features[image_category]['features'][indices]
    all_features[image_category]['group'] = all_features[image_category]['group'][indices]


# In[7]:


for image_category in all_features:
    if image_category not in group:
        continue
    print(image_category,
          len(all_features[image_category]['features']),
          len(all_features[image_category]['group']))


# In[8]:


# p_train = 0.75
# p_val = 0.125
p_train = 0.8
p_val = 0.1

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []
for image_category in all_features:
    if image_category not in group:
        continue

    n = len(all_features[image_category]['features'])
    n_train = int(n * p_train)
    x_train.append(all_features[image_category]['features'][:n_train])
    y_train.append(all_features[image_category]['group'][:n_train])

    n_val = int(n * p_val)
    x_val.append(all_features[image_category]['features'][n_train:n_train+n_val])
    y_val.append(all_features[image_category]['group'][n_train:n_train+n_val])

    x_test.append(all_features[image_category]['features'][n_train+n_val:])
    y_test.append(all_features[image_category]['group'][n_train+n_val:])

x_train = torch.cat(x_train).to(device)
y_train = torch.cat(y_train).to(device)
x_val = torch.cat(x_val).to(device)
y_val = torch.cat(y_val).to(device)
x_test = torch.cat(x_test).to(device)
y_test = torch.cat(y_test).to(device)


# In[9]:


# x_train, x_val, x_test = [], [], []
# y_train, y_val, y_test = [], [], []

# x_train = [all_features['gender_base_train']['features'], all_features['neg_gender_base_train']['features']]
# y_train = [all_features['gender_base_train']['group'], all_features['neg_gender_base_train']['group']]

# x_val = [all_features['gender_base_val']['features'], all_features['neg_gender_base_val']['features']]
# y_val = [all_features['gender_base_val']['group'], all_features['neg_gender_base_val']['group']]

# x_test = [all_features['gender_base_test']['features'], all_features['neg_gender_base_test']['features']]
# y_test = [all_features['gender_base_test']['group'], all_features['neg_gender_base_test']['group']]

# x_train = torch.cat(x_train).to(device)
# y_train = torch.cat(y_train).to(device)
# x_val = torch.cat(x_val).to(device)
# y_val = torch.cat(y_val).to(device)
# x_test = torch.cat(x_test).to(device)
# y_test = torch.cat(y_test).to(device)


# In[10]:


x_train.shape, x_val.shape, x_test.shape


# In[11]:


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# In[57]:


from torch import nn
n_features = 28
n_targets = 5
hidden_sizes = [16,14]
prev_hidden_size = hidden_sizes.pop(0)
layers = [nn.Linear(n_features, prev_hidden_size)]
for hidden_size in hidden_sizes:
    layers.append(nn.Linear(prev_hidden_size, hidden_size))
    prev_hidden_size = hidden_size
layers.append(nn.Linear(prev_hidden_size, n_targets))
layers


# In[12]:


from torch import nn
from torch.optim import SGD
from sklearn.metrics import precision_recall_fscore_support

from collections import defaultdict
from torch.nn import functional as F
from tqdm.auto import tqdm
import copy


def create_linear_probe(hidden_size):
    if isinstance(hidden_size, list):
        hidden_sizes = hidden_size
    else:
        hidden_sizes = [hidden_size]

    n_features = x_train.shape[1]
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
            optimizer.zero_grad()

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


# In[13]:


from itertools import product

is_binary_task = len(set(group.values())) == 2
relevant_metric_name = 'f1' if is_binary_task else 'acc'

all_metrics = {}
grid_best_relevant_metric = 0
grid_best_model = None
grid_best_metrics = None
best_hidden_size = None
best_lr = None
for exp_idx, (hidden_size, lr) in enumerate(product([32, 64, 128, 256],
                                                    [1e-1, 1e-2, 1e-3]), start=1):

    print(f'Start experiment: {exp_idx}')

    linear_probe = create_linear_probe(hidden_size=hidden_size)
    optimizer = SGD(linear_probe.parameters(), lr=lr, momentum=0.9)

    best_model, metrics, best_relevant_metric = train(linear_probe,
                                                      optimizer,
                                                      train_loader,
                                                      val_loader,
                                                      num_epochs=500,
                                                      patience=20,
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


# In[ ]:





# In[14]:


is_binary_task = len(set(group.values())) == 2
criterion = F.binary_cross_entropy_with_logits if is_binary_task else F.cross_entropy

grid_best_model.eval()
cum_loss = 0
cum_logit_pred = []
cum_y = []
for x, y in tqdm(test_loader):
    x, y = x.to(device), y.to(device)
    bsz = x.shape[0]
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


# In[15]:


print('test_acc: ', grid_best_metrics['test_acc'])
if is_binary_task:
    print('test_prec', grid_best_metrics['test_prec'])
    print('test_recall', grid_best_metrics['test_recall'])
    print('test_f1', grid_best_metrics['test_f1'])


# In[16]:




# In[22]:


track_metric = 'test_f1' if is_binary_task else 'test_acc'
best_test_metrics = defaultdict(list)


for _ in range(5):
    test_metrics = {}
    hidden_size, lr = best_hidden_size, best_lr
    linear_probe = create_linear_probe(hidden_size=hidden_size)
    optimizer = SGD(linear_probe.parameters(), lr=lr, momentum=0.9)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    best_model, metrics, best_relevant_metric = train(linear_probe,
                                                             optimizer,
                                                             train_loader,
                                                             val_loader,
                                                             num_epochs=500,
                                                             patience=10,
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


# In[23]:


import numpy as np
print(model_to_use)
print(group_name)
for metric_name in best_test_metrics:
    print(f'mean {metric_name}: {np.mean(best_test_metrics[metric_name]):.4f}')
    print(f'std {metric_name}: {np.std(best_test_metrics[metric_name]):.4f}')
    print(best_test_metrics[metric_name])


# In[ ]:





# In[6]:


result_path = 'outputs/results/skill-probe/results.pth'
try:
    all_results = torch.load(result_path)
except FileNotFoundError:
    all_results = []

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


# In[ ]:





# ### Analize Results

# In[7]:


[(r['hidden_size'], r['lr']) for r in all_results]


# In[8]:


# result_path = 'outputs/results/skill-probe/results.pth'
# all_results = torch.load(result_path)


# In[16]:


model_names = {
    'caption_base_flickr': 'base',
    'caption_flickr_augmented_c': 'color',
    'caption_augmented_flickr': 'gender',
    'caption_flickr_augmented_counting': 'counting',
    'caption_flickr_augmented_color+counting+gender': 'all',
}

to_float_result = lambda l_: '{:.1f}'.format(float(np.mean(l_)*100))

processed_results = [
   [model_names[r['model']], r['skill'],
    to_float_result(r['test_metrics']['test_f1']),
    to_float_result(r['test_metrics']['test_prec']),
    to_float_result(r['test_metrics']['test_recall'])]  for r in all_results
]


# In[17]:


import pandas as pd

df = pd.DataFrame(processed_results)
df.columns = ['model', 'skill', 'f1-score', 'precision', 'recall']


# In[18]:


# df.style.highlight_max(color = 'lightgreen', axis = 0)
# df.style.highlight_max("font-weight: bold", axis=0)
# def df_style(val):
#     return "font-weight: bold"
# df.style.highlight_max(df_style, subset=last_row)


# In[19]:


df


# In[23]:


color_df = df[df['skill'] == 'color']
color_df


# In[24]:


counting_df = df[df['skill'] == 'counting']
counting_df


# In[25]:


gender_df = df[df['skill'] == 'gender']
gender_df


# In[48]:


# combined_df = color_df.merge(counting_df, how='inner', on='model', suffixes=('_color', '_counting'))
# combined_df = combined_df.merge(gender_df, how='inner', on='model', suffixes=('_combined', '_gender'))

combined_df = df.pivot('model', 'skill').reorder_levels([1, 0], axis=1).sort_index(axis=1)


# In[52]:


combined_df


# In[51]:


print(combined_df.to_latex())


# In[ ]:




