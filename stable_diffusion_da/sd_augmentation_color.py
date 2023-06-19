import pandas as pd 
import ast
import torch
from diffusers import StableDiffusionPipeline
import os
from IPython.display import Image, display
import random 
from tqdm import tqdm
from functools import partialmethod

random.seed(10)

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16"
)

pipeline = pipeline.to("cuda")


def generate_images(
    prompt,
    num_images_to_generate,
    image_id,
    num_images_per_prompt=1,
    guidance_scale=8,
    display_images=False
):

    num_iterations = num_images_to_generate // num_images_per_prompt
    
    for i in range(num_iterations):
        images = pipeline(
            prompt, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale
        )
        for idx, image in enumerate(images.images):
            image_name = f"images/image_{(i*num_images_per_prompt)+idx}_{image_id}_color.png"
            image.save(image_name)
            if display_images:
                display(Image(filename=image_name, width=128, height=128))

df_gender_captions = pd.read_csv('data/Image_quality_testing_color.tsv', sep='\t')
df_gender_captions['augmented_captions'] = [ast.literal_eval(x) for x in df_gender_captions['augmented_captions']]

captions = []

for x, y, z in zip(df_gender_captions['image_ID'], df_gender_captions['caption_ID'], df_gender_captions['augmented_captions']) :
    id_ = str(x) + '_' + str(y)
    captions.append([id_ , z[0]])



for id_ , capt in captions:
    generate_images(capt, 1, image_id= id_ ,guidance_scale=8, display_images=False) 



    