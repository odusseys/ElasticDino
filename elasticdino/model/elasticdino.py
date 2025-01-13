import torch
import torch.nn as nn
import math
from elasticdino.model.dino import DinoV2, resize_for_dino
from elasticdino.model.layers import ProjectionLayer, ResidualBlock, Activation


def make_base_locations(batch_size, size, dtype):
    x = torch.arange(size, device="cuda", dtype=dtype) * (2 / size) - 1
    y = torch.arange(size, device="cuda", dtype=dtype) * (2 / size) - 1
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
        *[torch.compile(ResidualBlock(n_features), dynamic=True) for _ in range(n_layers)]
    )

    last_layer = nn.Conv2d(n_features // 8, 2, 1)
    # initialize last layer to small values and no bias to have an initial field that is close to the identity
    torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.003, generator=None)
    nn.init.zeros_(last_layer.bias)

    self.deformer = torch.compile(nn.Sequential(
        nn.Conv2d(n_features, n_features // 2, 1),
        Activation(),
        nn.Conv2d(n_features // 2, n_features // 4, 1),
        Activation(),
        nn.Conv2d(n_features // 4, n_features // 8, 1),
        Activation(),
        last_layer,
    ), dynamic=True)


  def forward(self, features, image):
    image = self.image_encoder(image)
    f = self.feature_encoder(features)
    f = self.convs(torch.cat([f, image], dim=1))
    base_locations = make_base_locations(image.shape[0], image.shape[-1], dtype=image.dtype).permute((0, 3, 1, 2))
    field = base_locations + self.deformer(f)
    field = field.permute((0, 2, 3, 1))
    return torch.nn.functional.grid_sample(features, field, padding_mode="border", align_corners=False)

class ElasticDinoStage(nn.Module):
  def __init__(self, layer_config, n_features_in, n_image_features):
    super().__init__()
    self.blocks = nn.ModuleList([
        DeformerBlock(layer_config["layers_per_block"], layer_config["hidden_features"], n_features_in, n_image_features)
        for _ in range(layer_config["n_blocks"])
    ])

  def forward(self, features, images):
    images = torch.nn.functional.interpolate(images, features.shape[-1], mode="bilinear")
    for block in self.blocks:
      features = block(features, images)
    return features


CONFIGS = {
  "elasticdino-L-64":  dict(
        dino_model="l",
        n_features_in=1024,
        layers={
            64: dict(hidden_features=256, n_blocks=4, layers_per_block=8),
            128: dict(hidden_features=256, n_blocks=3, layers_per_block=8),
        },
        start_size=64,
        target_size=128,
    )
}

def repair_checkpoint(path):
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix) :]
        return text
    ckpt = torch.load(path)
    in_state_dict = ckpt
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
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
  def __init__(self, config):
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

    self.dino = DinoV2(config["dino_model"])

  def forward(self, images):
    features = self.dino.get_features_for_tensor(resize_for_dino(images, self.config["starting_size"]))
    images = nn.functional.interpolate(images, self.config["target_size"], mode="bilinear", antialias=True)
    n = len(self.stages)
    current_size = features.shape[-1]
    for i in range(n):
      features = self.stages[i](features, images)
      if i < n - 1:
        current_size *= 2
        features = torch.nn.functional.interpolate(features, current_size, mode="nearest")
    return features
  
  def from_pretrained(checkpoint_path, model_name):
    config = CONFIGS[model_name]
    repair_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = ElasticDino(config)
    model.load_state_dict(checkpoint)
    return model
