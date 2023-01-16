"""
Script of the dataloader class 
author: Valentin Barriere
12/22
"""

from datasets import load_dataset
import os 
import pandas as pd 
from tqdm import tqdm

str2list = lambda x: [k[1:-1] for k in x.strip('][').split(', ')]

class dataLoader():
    """
    Load the data for the captions from tsv files and the images from the dataset API  
    """
    def __init__(self, 
        path_text_data : str = './', 
        list_skills : list = ['gender', 'counting', 'color'], 
        debug_load_img : int = 0,
        ):
        """
        debug_load_img: for debugging purpose since its long to load all the images (4mn)
        """
        self.path_text_data = path_text_data
        self.list_skills = list_skills

        # text
        self.text_dataset = {'all' : pd.read_csv(path_text_data + 'Caption_all.tsv', sep='\t')}
        self.text_dataset.update({'test_%s' %skill : pd.read_csv(path_text_data + 'Caption_test_%s.tsv'%skill, sep='\t') for skill in list_skills if os.path.isfile(path_text_data + 'Caption_test_%s.tsv'%skill)})
        self.text_dataset.update({'train_%s' %skill : pd.read_csv(path_text_data + 'Caption_train_%s.tsv'%skill, sep='\t') for skill in list_skills if os.path.isfile(path_text_data + 'Caption_test_%s.tsv'%skill)})
        for k in self.text_dataset.keys():
            if 'train' in k:
                self.text_dataset[k].prompt_segmentation = self.text_dataset[k].prompt_segmentation.map(str2list)
        print("Text loaded:", list(self.text_dataset.keys()))

        # image
        self.img_dataset = load_dataset("carlosejimenez/flickr30k_images_SimCLRv2") # images 
        # capt2idx = {int(os.path.basename(img_dataset['train'][x]['image_id']).split('.')[0]) : x for x in range(len(img_dataset['train']))}
        
        self.idx2iid = {}
        if debug_load_img:
            print('WARNING: Just loading the first %d images'%debug_load_img)
            range_to_load = debug_load_img
        else:
            range_to_load = len(self.img_dataset['train'])
        for x in tqdm(range(range_to_load)):
            idx2iid = {int(os.path.basename(self.img_dataset['train'][x]['image_id']).split('.')[0]) : x}

        print("Image loaded")

    def iid2img(self, iid):
        """
        Function that return the image regarding the iid
        """
        return self.img_dataset['train'][self.idx2iid[iid]]['image']

