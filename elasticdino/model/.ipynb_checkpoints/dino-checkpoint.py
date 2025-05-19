
import torchvision.transforms as transforms
import torch
import math
import os

MINIMAL_IMAGE_SIZE = 224
MINIMAL_GRID_SIZE = MINIMAL_IMAGE_SIZE // 14


FEATURE_SIZES = {
    's': 384,
    'b': 768,
    'l': 1024,
    'g': 1536
}

def resize_for_dino(images, starting_size):
    factor = starting_size // MINIMAL_GRID_SIZE
    images = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True)
    return images  

import torch.nn as nn

class DinoWrapper(nn.Module):
  def __init__(self, dino):
    super().__init__()
    self.dino = dino
  
  def forward(self, x, n):
    return self.dino.get_intermediate_layers(x, n=n)

class DinoV2:
  def __init__(self, dino_model):
    dino_backbone = DinoWrapper(torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{dino_model}14_reg'))
    dino_backbone.requires_grad_(False)
    self.dino_backbone = torch.compile(dino_backbone.eval().to("cuda", torch.float32))
    self.feature_size = FEATURE_SIZES[dino_model]

  def prepare_images(self, images):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device="cuda") / 255.0
    return tensors

  def get_features_for_tensor(self, images):
    return self.get_intermediate_features_for_tensor(images, 1)[0]

  def get_intermediate_features_for_tensor(self, images, n):
    with torch.no_grad():
      res = self.dino_backbone(images, n=n)
      grid_size = int(math.sqrt(res[0].shape[1]))
      res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
      del images
      return res
  
  def get_features(self, images):
    images = self.prepare_images(images)
    return self.get_features_for_tensor(images)

