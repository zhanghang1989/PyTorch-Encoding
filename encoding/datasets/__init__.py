from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
