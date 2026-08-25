
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
  def __init__(self, dino_repo, dino_model, use_registers=True):
    super().__init__()
    # logger.info("Loading DinoV2 backbone")
    cache_key = (dino_repo, dino_model, use_registers)
    if cache_key in DINO_CACHE:
      dino_backbone = DINO_CACHE[cache_key]
    else:
      source = "github" if dino_repo == 'facebookresearch/dinov2' else "local"
      model_name = f'dinov2_vit{dino_model}14' + ('_reg' if use_registers else '')
      dino_backbone = torch.hub.load(dino_repo, model_name, source=source)
      DINO_CACHE[cache_key] = dino_backbone

    # logger.info("DinoV2 backbone loaded")
    dino_backbone.requires_grad_(False)
    self.dino_backbone = dino_backbone.eval()
    self.feature_size = FEATURE_SIZES[dino_model]

  def prepare_images(self, images, device="cuda"):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device=device) / 255.0
    return tensors

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # Doc: functions after addition of normalization
  def get_features_for_tensor(self, images):
    return self.get_intermediate_features_for_tensor(images, 1)[0]

  def get_features_and_class_token_for_tensor(self, images):
    with torch.no_grad():
      images = self._normalize_images(images)
      patch_tokens, class_token = self.dino_backbone.get_intermediate_layers(
        images,
        n=1,
        return_class_token=True,
      )[0]
      return self._reshape_patch_tokens(patch_tokens), class_token

  def _normalize_images(self, images):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    return transforms.Normalize(
      mean=IMAGENET_DEFAULT_MEAN,
      std=IMAGENET_DEFAULT_STD,
    )(images)

  def _reshape_patch_tokens(self, tokens):
    grid_size = int(math.sqrt(tokens.shape[1]))
    return tokens.reshape(
      tokens.shape[0], grid_size, grid_size, self.feature_size
    ).permute(0, 3, 1, 2)

  def get_intermediate_features_for_tensor(self, images, n):
    with torch.no_grad():
      images = self._normalize_images(images)
      res = self.dino_backbone.get_intermediate_layers(images, n=n)
      res = [self._reshape_patch_tokens(r) for r in res]
      del images
      return res
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # Doc: Original functions (without normalization)
  # def get_features_for_tensor(self, images):
  #   return self.get_intermediate_features_for_tensor(images, 1)[0]

  # def get_intermediate_features_for_tensor(self, images, n):
  #   with torch.no_grad():
  #     res = self.dino_backbone.get_intermediate_layers(images, n=n)
  #     grid_size = int(math.sqrt(res[0].shape[1]))
  #     res = [r.reshape((r.shape[0], grid_size, grid_size, self.feature_size)).permute((0, 3, 1, 2)) for r in res]
  #     del images
  #     return res
  
  def get_features(self, images):
    images = self.prepare_images(images)
    return self.get_features_for_tensor(images)
