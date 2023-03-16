import timm
import torch
import torch.nn as nn
import requests

from PIL import Image
from pprint import pprint

IMAGENET_21k_URL = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'
NUMBER_OF_CLASSES = 21841

class TimmEvaluator(nn.Module):
    def __init__(
        self,
        architecture: str ='convnext_large_in22k'
    ) -> None:
        super().__init__()
        self.model = timm.create_model(architecture, pretrained=True)
        self.model.eval()
        self.labels = requests.get(IMAGENET_21k_URL).text.strip().split('\n')
        self.k = NUMBER_OF_CLASSES
        self.transform = timm.data.create_transform(
            **timm.data.resolve_data_config(self.model.pretrained_cfg)
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        output = self.model(*args, **kwargs)[0]
        probabilities = torch.nn.functional.softmax(output, dim=0)
        
        return probabilities
    
    def evaluate(self, image, label):
        image_tensor = self.transform(image)
        probabilities = self(image_tensor.unsqueeze(0))
        values, indices = torch.topk(probabilities, self.k)
        results = {self.labels[idx]: val.item() for val, idx in zip(values, indices)}

        if type(label) == str:
            return results[label]
        elif type(label) == list:
            return [results[sublabel] for sublabel in label]

class Model:
    pass

if __name__ == "__main__":
    evaluator = TimmEvaluator()

    image_path = 'cristian.jpg'
    image = Image.open(image_path)

    score = evaluator.evaluate(image, "homo, man, human_being, human")
    print(score)