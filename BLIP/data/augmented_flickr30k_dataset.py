import os
import pandas as pd

from torch.utils.data import Dataset, ConcatDataset

from PIL import Image

from data.utils import pre_caption

str2list = lambda x: [k[1:-1] for k in x.strip('][').split(', ')]

class augmented_flickr30k(Dataset):
    """
    Load the data for the captions from tsv files and the images from the dataset API
    """
    def __init__(self, annotation, transform, split, max_words=30, prompt=''):
        self.annotation = annotation
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.prompt = prompt
        self.img_ids = {}

        # Remove invalid images
        original_len = len(self.annotation)
        # self.annotation = [ann for ann in self.annotation if os.path.exists(ann['image_path'])]
        valid_annotations = []
        for ann in self.annotation:
            try:
                Image.open(ann['image_path']).convert('RGB')
            except:
                continue
            valid_annotations.append(ann)

        self.annotation = valid_annotations
        print('{:.2f}% Images found and loaded.'.format(len(self.annotation) * 100 / original_len))

        n = 0
        for ann in self.annotation:
            img_id = ann['image_path']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = ann['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = ann['augmented_caption'][0] if 'augmented_caption' in ann else ann['caption']
        caption = self.prompt + pre_caption(caption, self.max_words)

        if 'train' in self.split:
            return image, caption, self.img_ids[ann['image_path']]
        else:
            if 'train_' in self.split:
                image_id = str(ann['image_ID']) + '_' + str(ann['caption_ID'])
                image_id = f'{self.split}_{image_id}'
            else:
                image_id = str(ann['image_ID'])
            return image, image_id

    @classmethod
    def build_all(cls,
                  path_text_data : str,
                  image_roots_dict : dict,
                  transforms : dict,
                  max_words=30,
                  prompt='',
                  list_skills : list = ['gender', 'counting', 'color'],
                  image_ext_dict : dict = None
                  ):
        # text
        text_dataset = {'all' : pd.read_csv(path_text_data + 'Caption_all.tsv', sep='\t')}
        text_dataset.update({'test_%s' %skill : pd.read_csv(path_text_data + 'Caption_testing_%s.tsv'%skill, sep='\t') for skill in list_skills if os.path.isfile(path_text_data + 'Caption_testing_%s.tsv'%skill)})
        text_dataset.update({'train_%s' %skill : pd.read_csv(path_text_data + 'Caption_training_%s.tsv'%skill, sep='\t') for skill in list_skills if os.path.isfile(path_text_data + 'Caption_training_%s.tsv'%skill)})

        for k in text_dataset.keys():
            if 'train' in k:
                text_dataset[k] = text_dataset[k][text_dataset[k]['sample']]
                text_dataset[k]['prompt_segmentation'] = text_dataset[k].prompt_segmentation.map(str2list)
                text_dataset[k]['augmented_captions'] = text_dataset[k].augmented_captions.map(str2list)

        print("Text loaded:", list(text_dataset.keys()))

        base_image_root, base_image_ext = image_roots_dict['all'], image_ext_dict['all']
        for k in text_dataset.keys():
            image_root = image_roots_dict[k] if k in image_roots_dict else base_image_root
            image_ext = image_ext_dict[k] if k in image_ext_dict else base_image_ext
            if 'train' in k:
                text_dataset[k]['image_path'] = (image_root
                                                #  + text_dataset[k].image_ID.astype(str)
                                                 + text_dataset[k].image_ID.astype(str) + '_' + text_dataset[k].caption_ID.astype(str)
                                                 + image_ext)
            else:
                text_dataset[k]['image_path'] = image_root + text_dataset[k].image_ID.astype(str) + image_ext

        text_dataset['train'] = text_dataset['all'][text_dataset['all'].split == 'train']
        text_dataset['val'] = text_dataset['all'][text_dataset['all'].split == 'val']
        text_dataset['test'] = text_dataset['all'][text_dataset['all'].split == 'test']
        del text_dataset['all']

        # Train augmented
        datasets = {}
        for split, annotation in text_dataset.items():
            print(f"Creating {split} dataset")
            annotation = annotation.to_dict(orient='records')
            datasets[split] = cls(
                annotation=annotation,
                transform=transforms[split],
                split=split,
                max_words=max_words,
                prompt=prompt,
            )

        return (ConcatDataset([ds for split, ds in datasets.items() if 'train' in split]),
                ConcatDataset([ds for split, ds in datasets.items() if 'val' in split]),
                ConcatDataset([ds for split, ds in datasets.items() if 'test' in split]))
