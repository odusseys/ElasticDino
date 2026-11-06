import torch
import torch.nn as nn
import math
from elasticdino.model.dino import DinoV2, resize_for_dino
from elasticdino.model.layers import ProjectionLayer, ResidualBlock, Activation, FCLayer
import logging
import json
import warnings
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


# logger = logging.getLogger("ElasticDino")

# Doc ? seems to not be used !!
class TaskHeads(nn.Module):
  def __init__(self, n_features_in, n_segmentation_classes):
    super().__init__()
    
    def make_head(dim):
      return nn.Sequential(
        nn.Conv2d(n_features_in, n_features_in // 2, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in // 2, n_features_in // 4, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in // 4, n_features_in // 8, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in // 8, dim, 1),
    )

    self.reproduction_head = make_head(3)

    self.depth_head = make_head(1)

    self.segmentation_head = nn.Sequential(
      nn.Conv2d(n_features_in, n_features_in, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in, n_features_in, 1),
        nn.ReLU(),
        nn.Conv2d(n_features_in, n_segmentation_classes, 1),
    )

    self.diffuse_illumination_head = make_head(3)
    self.diffuse_reflectance_head = make_head(3)
    self.residual_head = make_head(3)
    self.normal_bump_cam_head = make_head(3)

  def forward(self, x):
    return dict(
                depth=self.depth_head(x),
                segmentation=self.segmentation_head(x),
                reproduced=self.reproduction_head(x),
                diffuse_illuminations=self.diffuse_illumination_head(x),
                diffuse_reflectances=self.diffuse_reflectance_head(x),
                residuals=self.residual_head(x),
                normal_bump_cams=self.normal_bump_cam_head(x),
              )
  
def make_base_locations(batch_size, size, dtype, device):
    # Doc torch.arange is a PyTorch function that returns a 1D tensor with evenly spaced values within a specified range
    # These lines create range in [-1,1) (normalized coordinates)
    x = torch.arange(size, device=device, dtype=dtype) * (2 / size) - 1
    y = torch.arange(size, device=device, dtype=dtype) * (2 / size) - 1
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    res = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return res


class DeformerBlock(nn.Module):
  # Doc n_features_in is number of features in the input feature map and n_image_features is number of features in the input image
  def __init__(self, n_layers, n_features, n_features_in, n_image_features):
    super().__init__()
    self.image_encoder = ProjectionLayer(n_image_features, n_features)
    self.feature_encoder = ProjectionLayer(n_features_in, n_features)

    # Doc n_features * 2 because use it on concatenation of image and feature encoders
    self.convs = nn.Sequential(
        ProjectionLayer(n_features * 2, n_features),
        *[ResidualBlock(n_features) for _ in range(n_layers)]
    )

    last_layer = nn.Conv2d(n_features // 8, 2, 1)
    # initialize last layer to small values and no bias to have an initial field that is close to the identity
    torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.003, generator=None)
    nn.init.zeros_(last_layer.bias)

    self.deformer = nn.Sequential(
        nn.Conv2d(n_features, n_features // 2, 1),
        Activation(),
        nn.Conv2d(n_features // 2, n_features // 4, 1),
        Activation(),
        nn.Conv2d(n_features // 4, n_features // 8, 1),
        Activation(),
        last_layer,
    )


  def forward(self, features, image, return_displacements=False, additional_inputs=[]):
    image = self.image_encoder(image)
    f = self.feature_encoder(features)
    f = self.convs(torch.cat([f, image], dim=1))
    # Doc Permute operations: Convert between (B,H,W,2) ↔ (B,2,H,W) tensor layouts
    base_locations = make_base_locations(image.shape[0], image.shape[-1], image.dtype, image.device).permute((0, 3, 1, 2))
    # Doc Neural network predicts 2D displacement vectors (dx, dy) for each pixel
    displacements = self.deformer(f)
    field = base_locations + displacements
    field = field.permute((0, 2, 3, 1))
    # Doc grid_sample: Samples input tensors at deformed coordinates
    # Doc padding_mode="border": Uses edge values for out-of-bounds sampling
    results = dict(features=torch.nn.functional.grid_sample(features, field, padding_mode="border", align_corners=False))
    add = []
    for x in additional_inputs:
      add.append(torch.nn.functional.grid_sample(x, field, padding_mode="border", align_corners=False))
    results["additional_inputs"] = add
    if return_displacements:
      results["displacements"] = displacements
    return results


class ElasticDinoStage(nn.Module):
  def __init__(self, layer_config, n_features_in, n_image_features):
    super().__init__()
    self.blocks = nn.ModuleList([
        DeformerBlock(layer_config["layers_per_block"], layer_config["hidden_features"], n_features_in, n_image_features)
        for _ in range(layer_config["n_blocks"])
    ])

  def forward(self, features, images, return_displacements=False, additional_inputs=[]):
    displacements = []
    images = torch.nn.functional.interpolate(images, features.shape[-1], mode="bilinear")
    for block in self.blocks:
        block_results = block(features, images, return_displacements, additional_inputs)
        features = block_results["features"]
        additional_inputs = block_results["additional_inputs"]
        if return_displacements:
          displacements.append(block_results["displacements"])
    results = dict(features=features)
    if return_displacements:
      results["displacements"] = displacements
    if additional_inputs is not None:
      results["additional_inputs"] = additional_inputs
    return results


def prepare_image_input(images, device):
    """Convert an RGB PIL image or CHW/BCHW tensor into a square float batch."""
    if isinstance(images, Image.Image):
        images = pil_to_tensor(images.convert("RGB"))
    if not isinstance(images, torch.Tensor):
        raise TypeError("Expected a PIL image or a CHW/BCHW tensor")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected an RGB tensor with shape (3, H, W) or (B, 3, H, W)")
    if images.dtype == torch.uint8:
        images = images.to(device=device, dtype=torch.float32) / 255.0
    elif images.is_floating_point():
        images = images.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("Tensor images must be uint8 or floating point")

    height, width = images.shape[-2:]
    if height != width:
        warnings.warn("Input image is not square and will be center cropped.", UserWarning, stacklevel=2)
        size = min(height, width)
        top, left = (height - size) // 2, (width - size) // 2
        images = images[..., top:top + size, left:left + size]
    return images


class ElasticDino(nn.Module):
  def __init__(self, config, dino_repo):
    super().__init__()
    self.config = config

    n_features_in = config["n_features_in"]
    layer_configs = config["layers"]

    # Doc probably 3 for RGB
    n_image_features = 3
    n_upscales = int(math.log2(config["target_size"] // config["start_size"])) + 1
    assert n_upscales == len(layer_configs), "Incompatible resolutions and feature config"

    self.stages = nn.ModuleList([
        ElasticDinoStage(layer_configs[res], n_features_in, n_image_features)
        for res in sorted(layer_configs, key=int)
    ])

    self.dino = DinoV2(
      dino_repo,
      config["dino_model"],
      use_registers=config.get("use_registers", True),
    )

  def forward(self, images, return_all_scales=False, return_original_features=False, return_displacements=False, n_hidden_layers=None, return_class_token=False):
    images = prepare_image_input(images, next(self.stages.parameters()).device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=images.is_cuda):
      if n_hidden_layers is None:
        additional_inputs = []
        dino_images = resize_for_dino(images, self.config["start_size"])
        if return_class_token:
          features_in, class_token = self.dino.get_features_and_class_token_for_tensor(dino_images)
        else:
          features_in = self.dino.get_features_for_tensor(dino_images)
      else:
        if return_class_token:
          raise ValueError("return_class_token is not supported with n_hidden_layers")
        additional_inputs = self.dino.get_intermediate_features_for_tensor(resize_for_dino(images, self.config["start_size"]), n_hidden_layers)
        features_in = additional_inputs.pop(-1)
      features = features_in
      images = nn.functional.interpolate(images, self.config["target_size"], mode="bilinear", antialias=True)
      results = []
      all_displacements = []
      n = len(self.stages)
      current_size = features.shape[-1]
      for i in range(n):
        stage_outputs = self.stages[i](features, images, return_displacements, additional_inputs)
        features = stage_outputs["features"]

        if return_all_scales:
          results.append(features)
        additional_inputs = stage_outputs["additional_inputs"]

        if return_displacements:
          all_displacements.append(stage_outputs["displacements"])
        del stage_outputs
        if i < n - 1:
          current_size *= 2
          features = torch.nn.functional.interpolate(features, current_size, mode="nearest")
      
      if (not return_all_scales) and (not return_original_features) and (not return_displacements) and n_hidden_layers is None and (not return_class_token):
        return features

      out = dict(deformed_features=features)
      if return_all_scales:
        out["all_scales"] = results
      if return_original_features:
        out["original_features"] = features_in
      if return_displacements:
        out["displacements"] = all_displacements
      if n_hidden_layers is not None:
        additional_inputs.append(features)
        out["hidden_layers"] = additional_inputs
      if return_class_token:
        out["class_token"] = class_token
      
      del additional_inputs
      del all_displacements
      del features_in
      del results
      del features
      del images
      return out
    

  def parameters(self):
    return self.stages.parameters()
  
  def train(self, value=True):
    self.stages.train(value)
    return self
  
  @classmethod
  def from_pretrained(cls, repo_id, *, revision=None, token=None):
    """Load architecture and weights from a Hugging Face model repository."""
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, token=token)
    with open(config_path, encoding="utf-8") as config_file:
      config = json.load(config_file)
    checkpoint_path = hf_hub_download(
      repo_id=repo_id, filename=config["checkpoint_file"], revision=revision, token=token
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    checkpoint = {key.replace("_orig_mod.", ""): value for key, value in checkpoint.items()}
    model = cls(config, config["dino_repo"])
    model.stages.load_state_dict({key.removeprefix("stages."): value for key, value in checkpoint.items()})
    return model
