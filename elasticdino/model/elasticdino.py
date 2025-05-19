import torch
import torch.nn as nn
import math
from elasticdino.model.dino import DinoV2, resize_for_dino
from elasticdino.model.layers import ProjectionLayer, ResidualBlock, Activation, FCLayer
import logging

# logger = logging.getLogger("ElasticDino")

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
    x = torch.arange(size, device=device, dtype=dtype) * (2 / size) - 1
    y = torch.arange(size, device=device, dtype=dtype) * (2 / size) - 1
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    res = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return res


class DeformerBlock(nn.Module):
  def __init__(self, n_layers, n_features, n_features_in, n_image_features):
    super().__init__()
    self.image_encoder = ProjectionLayer(n_image_features, n_features)
    self.feature_encoder = ProjectionLayer(n_features_in, n_features)

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
    base_locations = make_base_locations(image.shape[0], image.shape[-1], image.dtype, image.device).permute((0, 3, 1, 2))
    displacements = self.deformer(f)
    field = base_locations + displacements
    field = field.permute((0, 2, 3, 1))
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


CONFIGS = {
  "elasticdino-64-L":  dict(
        dino_model="l",
        n_features_in=1024,
        layers={
            64: dict(hidden_features=256, n_blocks=4, layers_per_block=8),
            128: dict(hidden_features=256, n_blocks=3, layers_per_block=8),
        },
        start_size=64,
        target_size=128,
    ),
    "elasticdino-32-L": dict(
        dino_model="l",
        n_features_in=1024,
        layers={
            32: dict(hidden_features=512, n_blocks=5, layers_per_block=8),
            64: dict(hidden_features=256, n_blocks=4, layers_per_block=8),
            128: dict(hidden_features=256, n_blocks=3, layers_per_block=8),
        },
        start_size=32,
        target_size=128,
    ),
}

def repair_checkpoint(path):
    ckpt = torch.load(path, weights_only=True, map_location=torch.device("cpu"))
    in_state_dict = ckpt
    pairings = [
        (src_key, "".join(src_key.split("_orig_mod.")))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt = out_state_dict
    torch.save(ckpt, path)


class ElasticDino(nn.Module):
  def __init__(self, config, dino_repo):
    super().__init__()
    self.config = config

    n_features_in = config["n_features_in"]
    layer_configs = config["layers"]

    n_image_features = 3
    n_upscales = int(math.log2(config["target_size"] // config["start_size"])) + 1
    assert n_upscales == len(layer_configs), "Incompatible resolutions and feature config"

    self.stages = nn.ModuleList([
        ElasticDinoStage(layer_configs[res], n_features_in, n_image_features) for res in layer_configs
    ])

    self.dino = DinoV2(dino_repo, config["dino_model"])

  def forward(self, images, return_all_scales=False, return_original_features=False, return_displacements=False, n_hidden_layers=None):
    if n_hidden_layers is None:
      additional_inputs = []
      features_in = self.dino.get_features_for_tensor(resize_for_dino(images, self.config["start_size"]))
    else:
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

      del stage_outputs

      if return_displacements:
        all_displacements.append(features["displacements"])
      if i < n - 1:
        current_size *= 2
        features = torch.nn.functional.interpolate(features, current_size, mode="nearest")
    
    if (not return_all_scales) and (not return_original_features) and (not return_displacements) and n_hidden_layers is None:
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
  
  def from_pretrained(checkpoint_path, model_name, dino_repo='facebookresearch/dinov2'):
    # logger.info("Loading ElasticDino", model_name)
    config = CONFIGS[model_name]
    repair_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = ElasticDino(config, dino_repo)
    # don't load parameters in the pretrained dino
    tmp_dino = model.dino
    model.dino = None
    model.load_state_dict(checkpoint)
    model.dino = tmp_dino
    # logger.info("ElasticDino loaded successfully")
    return model
