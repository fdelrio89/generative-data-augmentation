import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms

# model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
# model.eval()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Create predictor
predictor = DefaultPredictor(cfg)

image_path = 'coco.png'
image = Image.open(image_path)

# Define image transforms
# transforms = T.AugmentationList([
#     T.ColorTransform,
#     ResizeTransform(image.size[0], image.size[1]),
# ])

# tensor_image, _ = apply_image_transforms(image, transforms)

# inputs = [{
#     "image": tensor_image
# }]

with torch.no_grad():
  # Run inference on input image
  outputs = predictor(image)
  print(outputs)