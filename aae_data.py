import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast
import re
from typing import Union, List

flickr_path = "carlosejimenez/flickr30k_images_SimCLRv2"

def remove_caption_punctuation(caption: str):
    """
    'hello .my.friend' -> 'hello myfriend'
    """
    symbol_to_keep = "-'"
    pattern = f"[^\w\s{re.escape(symbol_to_keep)}]"

    return re.sub(pattern, "", caption).lower()

def get_caption_tokens(element):
    caption = element["caption"]
    caption_tokens = remove_caption_punctuation(caption).split()
    prompt_segmentations = element["prompt_segmentation"]
    split_prompt_segmentations = [remove_caption_punctuation(prompt).split() for prompt in prompt_segmentations] 

    # we sum 1 to the index since AaE notation doesn't start at 0 (wtf)
    try:
        caption_tokens = [[caption_tokens.index(token) + 1 for token in tokens] for tokens in split_prompt_segmentations]
    except ValueError:
        caption_tokens = []

    element["caption_tokens"] = caption_tokens
    return element

def cast_element(element):
    element["augmented_captions"] = ast.literal_eval(element["augmented_captions"])
    element["word_detected"] = ast.literal_eval(element["word_detected"])
    element["prompt_segmentation"] = ast.literal_eval(element["prompt_segmentation"])
    return element

def process_aae_flickr_element(element):
    element = cast_element(element)
    element = get_caption_tokens(element)
    return element


class AaEFlickrDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, List[str]] = [
            "./data/Caption_training_color.tsv",
            "./data/Caption_training_gender.tsv",
            "./data/Caption_training_counting.tsv"
        ],
        split: str = "train"
    ) -> None:
        captions_dataset = self.load_and_process_captions(data_path, split)
        self.captions = captions_dataset[split]

    def load_and_process_captions(self, data_path, split):
        print("Processing captions...")
        if isinstance(data_path, str):
            data_path = [data_path]
        captions_dataset = datasets.load_dataset(
            "csv",
            data_files=data_path,
            delimiter="\t"
        )
        original_num_rows = captions_dataset.num_rows[split]
        original_num_columns = captions_dataset.num_columns[split]

        # Filter useful rows
        captions_dataset = captions_dataset.filter(lambda el: el['sample'])
        # Rename rows
        captions_dataset = captions_dataset.rename_column("image_ID", "image_id")
        captions_dataset = captions_dataset.rename_column("caption_ID", "caption_id")
        # Remove columns
        captions_dataset = captions_dataset.remove_columns(["sample"])
        # Cast augmented_captions to useful format and enerate new columns
        captions_dataset = captions_dataset.map(lambda element: process_aae_flickr_element(element))
        # filter the ones that raised ValueError
        captions_dataset = captions_dataset.filter(lambda element: len(element["caption_tokens"]) > 0)

        new_num_rows = captions_dataset.num_rows[split]
        new_num_columns = captions_dataset.num_columns[split]

        print(f"Reduced the number of rows from {original_num_rows} to {new_num_rows}")

        return captions_dataset

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        element = self.captions[idx]

        return {
            "augmented_captions": element["augmented_captions"],
            "caption_tokens": element["caption_tokens"]
        }

if __name__ == "__main__":
    dset = AaEFlickrDataset()
    for idx, i in enumerate(dset):
        print(i)
    print("All good! Dataloader is working properly :)")