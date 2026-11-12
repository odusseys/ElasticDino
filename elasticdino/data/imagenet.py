import torchvision.transforms.functional
from datasets import  load_from_disk
from torch.utils.data import DataLoader
import torchvision
import os
import torch

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 1))

def process(sample, image_size):
    img = sample["image"].convert("RGB") 
    l = min(img.width, img.height)
    img = img.crop((0, 0, l, l))
    img = img.resize((image_size, image_size))
    return torchvision.transforms.functional.pil_to_tensor(img) / 255.0

def collate_fn(image_size):
    def collate(samples):
        return torch.stack([process(s, image_size) for s in samples])
    return collate

CACHE = {}

def get_imagenet(path):
  if path in CACHE:
    return CACHE[path]
  else:
    imagenet = load_from_disk(path)
    CACHE[path] = imagenet
    return imagenet
   

def load_imagenet(path, batch_size, image_size=256, num_workers=None):
  NUM_WORKERS if num_workers is None else num_workers
  print("Loading imagenet")
  imagenet = get_imagenet(path)
  print("Imagenet loaded")
  return DataLoader(imagenet, batch_size=batch_size, collate_fn=collate_fn(image_size), shuffle=True, num_workers=NUM_WORKERS)
  