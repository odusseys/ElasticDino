
import torchvision.transforms as transforms
import torch
import math
import os
import logging

# logger = logging.getLogger("ElasticDino")

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

DINO_CACHE = {}

class DinoV2(nn.Module):
  def __init__(self, dino_repo, dino_model):
    super().__init__()
    # logger.info("Loading DinoV2 backbone")
    if dino_model in DINO_CACHE:
      dino_backbone = DINO_CACHE[dino_model]
    else:
      source = "github" if dino_repo == 'facebookresearch/dinov2' else "local"
      dino_backbone = torch.hub.load(dino_repo, f'dinov2_vit{dino_model}14_reg', source=source)
      DINO_CACHE[dino_model] = dino_backbone

    # logger.info("DinoV2 backbone loaded")
    dino_backbone.requires_grad_(False)
    self.dino_backbone = dino_backbone.eval()
    self.feature_size = FEATURE_SIZES[dino_model]

  def prepare_images(self, images, device="cuda"):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device=device) / 255.0
    return tensors

  def get_features_for_tensor(self, images):
    return self.get_intermediate_features_for_tensor(images, 1)[0]

  def get_intermediate_features_for_tensor(self, images, n):
    with torch.no_grad():
      res = self.dino_backbone.get_intermediate_layers(images, n=n)
      grid_size = int(math.sqrt(res[0].shape[1]))
      res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
      del images
      return res
  
  def get_features(self, images):
    images = self.prepare_images(images)
    return self.get_features_for_tensor(images)


