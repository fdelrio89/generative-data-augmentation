# Generative Data Augmentation (?)

## Synthetic Data Evaluator

metric should/could be for a parallel task:
- Object recognition
- number of objects being detected, color, gender, and the confidence of the classification 

They could also be to asses if it is a fake or real image

Model references:

- [InternImage](https://github.com/opengvlab/internimage)
- [detrx](https://github.com/IDEA-Research/detrex)
    - [visualize raw annotations](https://detrex.readthedocs.io/en/latest/tutorials/Tools.html)
- [torchvision](https://pytorch.org/vision/stable/models.html)
- [HF: timm](https://github.com/huggingface/pytorch-image-models)
- [Detectron2](https://github.com/facebookresearch/detectron2)
    -[Model usage](https://detectron2.readthedocs.io/en/latest/tutorials/models.html)
    -[List of pre-trained models](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
    -[Model zoo from python](https://detectron2.readthedocs.io/en/latest/modules/model_zoo.html)




AndreS:
- Usar clip con una imagen controlada, con el label persona
---
## El problema

3 categorías en las cuales hacemos augmentation:

    - Color
    - Género
    - Counting

Por ejemplo lo que hacemos, es que dado un caption tipo:

"The man in a green shirt was riding a bike"

Cambiamos "green" por "blue", y generamos una imagen ($i$) con el nuevo prompt $p$.

Queremos una wea que me evalúe si $i$ dado $p$ tiene sentido, o parece real.    

