import timm
import torch
import torch.nn as nn
import requests

from detectron2 import model_zoo
from PIL import Image
from pprint import pprint
from torchvision import transforms

IMAGENET_21k_URL = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'
NUMBER_OF_CLASSES = 21841

class TimmEvaluator(nn.Module):
    """
    Interface to evaluate synthetic images with PyTorch Image Models

    Go over https://timm.fast.ai/models to see available models.
    """

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
    
    def evaluate(self, image, label = None):
        assert label is not None, "you need Imagenet 21k label/s for this evaluator"

        image_tensor = self.transform(image)
        probabilities = self(image_tensor.unsqueeze(0))
        values, indices = torch.topk(probabilities, self.k)
        results = {self.labels[idx]: val.item() for val, idx in zip(values, indices)}

        if type(label) == str:
            return results[label]
        elif type(label) == list:
            return [results[sublabel] for sublabel in label]


class DetectronEvaluator:
    """
    Interface to evaluate synthetic images with Facebook's Detectron2

    Go over https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md to see available models.
    """

    def __init__(
        self,
        architecture: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    ) -> None:
        self.model = model_zoo.get(architecture, trained=True)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    @torch.no_grad()
    def evaluate(self, image):
        tensor_image = self.transform(image)
        inputs = [{"image": tensor_image}]
        outputs = self.model(inputs)

        return outputs


class ClipEvaluator:

    def __init__(self):
        pass

    def evaluate(self):
        pass

class DummyModel:
    pass

if __name__ == "__main__":
    evaluator = TimmEvaluator()

    image_path = 'cristian.jpg'
    image = Image.open(image_path)

    # score = evaluator.evaluate(image, "homo, man, human_being, human")
    score = evaluator.evaluate(image)
    print(score)