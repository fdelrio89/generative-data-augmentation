import timm
import torch
import requests

from PIL import Image
from pprint import pprint

model = timm.create_model('convnext_large_in22k', pretrained=True)
model.eval()
transform = timm.data.create_transform(
    **timm.data.resolve_data_config(model.pretrained_cfg)
)

image_path = 'cristian.jpg'
image = Image.open(image_path)
image_tensor = transform(image)

output = model(image_tensor.unsqueeze(0))
print(output.size())
probabilities = torch.nn.functional.softmax(output[0], dim=0)
values, indices = torch.topk(probabilities, 5)


IMAGENET_21k_URL = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'
IMAGENET_21k_LABELS = requests.get(IMAGENET_21k_URL).text.strip().split('\n')
# label = "homo, man, human_being, human"
# results = {IMAGENET_21k_LABELS[idx]: val.item() for val, idx in zip(values, indices)}
# print(results[label])
pprint([{'label': IMAGENET_21k_LABELS[idx], 'value': val.item()} for val, idx in zip(values, indices)])