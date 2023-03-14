import torch

from detectron2 import model_zoo
from PIL import Image
from torchvision import transforms

model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
model.eval()

image_path = 'img00.jpg'
image = Image.open(image_path)
transform = transforms.Compose([transforms.ToTensor()])
tensor_image = transform(image)

inputs = [{
    "image": tensor_image
}]

with torch.no_grad():
  outputs = model(inputs)
  print(outputs)