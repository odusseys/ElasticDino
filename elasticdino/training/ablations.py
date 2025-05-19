from elasticdino.model.dino import DinoV2, resize_for_dino
from elasticdino.model.layers import ProjectionLayer, ResidualBlock
import torch.nn as nn
import torch
from elasticdino.training.util import debug_features
from elasticdino.model.elasticdino import ElasticDino
from elasticdino.training.losses import reproduction_loss
from elasticdino.training.depth.train_depth import si_log_loss
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
import numpy as np
from elasticdino.training.depth.layers import UNet, UNet2, MyUNet


class AblateDeformationsStage(nn.Module):
  def __init__(self, layer_config, n_features_in, n_image_features):
    super().__init__()
    self.blocks = nn.Sequential(
      ProjectionLayer(n_features_in + n_image_features, layer_config["hidden_features"]),
      *[ResidualBlock(layer_config["hidden_features"]) for _ in range(layer_config["n_blocks"] * layer_config["layers_per_block"])],
      ProjectionLayer(layer_config["hidden_features"], n_features_in),
    )

  def forward(self, features, images):
    images = torch.nn.functional.interpolate(images, features.shape[-1], mode="bilinear")
    features = torch.cat([features, images], dim=1)
    return self.blocks(features)

import math

class AblateDeformations(nn.Module):
  def __init__(self, config, dino_repo):
    super().__init__()
    self.config = config

    n_features_in = config["n_features_in"]
    layer_configs = config["layers"]

    n_image_features = 3
    n_upscales = int(math.log2(config["target_size"] // config["start_size"])) + 1
    assert n_upscales == len(layer_configs), "Incompatible resolutions and feature config"

    self.stages = nn.ModuleList([
        AblateDeformationsStage(layer_configs[res], n_features_in, n_image_features) for res in layer_configs
    ])

    self.dino = DinoV2(dino_repo,config["dino_model"], )

  def forward(self, images, return_all_scales=False, return_original_features=False):
    features_in = self.dino.get_features_for_tensor(resize_for_dino(images, self.config["start_size"]))
    features = features_in    
    images = nn.functional.interpolate(images, self.config["target_size"], mode="bilinear", antialias=True)
    n = len(self.stages)
    current_size = features.shape[-1]
    results = []
    for i in range(n):
      features = self.stages[i](features, images)
      if return_all_scales:
        results.append(features)
      if i < n - 1:
        current_size *= 2
        features = torch.nn.functional.interpolate(features, current_size, mode="nearest")
    if return_all_scales:
      out = results
    else:
      out = features
    if return_original_features:
      return out, features_in
    return out

  def parameters(self):
    return self.stages.parameters()
  
  def train(self, x=True):
    self.stages.train(x)


class AblationTaskHeads(nn.Module):
  def __init__(self, n_features_in, tasks):
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

    self.heads = nn.ModuleDict({t: make_head(3) for t in tasks})


  def forward(self, x):
    return {t: self.heads[t](x) for t in self.heads}


class HypersimTaskHeads(nn.Module):
  def __init__(self, n_features_in):
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
    self.diffuse_illumination_head = make_head(3)
    self.diffuse_reflectance_head = make_head(3)
    self.residual_head = make_head(3)
    self.normal_bump_cam_head = make_head(3)

  def forward(self, x):
    return dict(
                reproduced=self.reproduction_head(x),
                diffuse_illuminations=self.diffuse_illumination_head(x),
                diffuse_reflectances=self.diffuse_reflectance_head(x),
                residuals=self.residual_head(x),
                normal_bump_cams=self.normal_bump_cam_head(x),
              )

HYPERSIM_TASKS = ["reproduced", "diffuse_illuminations", "diffuse_reflectances", "residuals", "normal_bump_cams"]

def ablation_loss(results_list, batch, features, min_scale, use_blur):
  rpr = 0
  diffl = 0
  diffr = 0
  resl = 0
  norml = 0
  bll = 0
  n = 0

  if use_blur:
      features = torch.nn.functional.interpolate(features, scale_factor=0.5, mode="bilinear")

  for r, f in results_list:
    resolution = batch["images"].shape[-1]
    scale = 1.0
    while resolution >= min_scale:
      n += scale
      # Check if key exists in dictionary before attempting to access it
      rpr += scale * reproduction_loss(r.get("reproduced"), batch["images"], resolution, order=0) if "reproduced" in r else 0
        
      diffl += scale * reproduction_loss(r.get("diffuse_illuminations"), batch["diffuse_illuminations"], resolution, order=2) if "diffuse_illuminations" in r else 0
        
      diffr += scale * reproduction_loss(r.get("diffuse_reflectances"), batch["diffuse_reflectances"], resolution, order=2) if "diffuse_reflectances" in r else 0
        
      resl += scale * reproduction_loss(r.get("residuals"), batch["residuals"], resolution, order=2) if "residuals" in r else 0
        
      norml += scale * reproduction_loss(r.get("normal_bump_cams"), batch["normal_bump_cams"], resolution, order=2) if "normal_bump_cams" in r else 0
      resolution //= 2
      scale *= 2
        
    if use_blur:
      downscaled = torch.nn.functional.interpolate(f, features.shape[-1], mode="bilinear") 
      bll += (downscaled - features).abs().mean()
      
    del r

  rpr = rpr / n
  diffl = diffl / n
  diffr = diffr / n
  resl = resl / n
  norml = norml / n
  bll = bll / len(results_list)


  return rpr + diffl + diffr + resl + norml + bll 
  
  
def debug_step(run_folder, batch, features, results, running_loss, n):
  with torch.no_grad():
    res = debug_features(features[0], [results[-1][0]])
    img = torchvision.transforms.functional.to_pil_image(batch["images"][0]).resize((128, 128))
    debug_image = Image.fromarray(np.hstack([img, res]))
    debug_image.save(f"{run_folder}/images/{n}.jpg")
    line = f"{n} {running_loss}"
    print(line)
    with open(f"{run_folder}/training_loss.txt", "a+") as f:
      f.write(line + "\n")

from datetime import datetime
import os

def init_run(project_folder):
  current_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
  run_folder = os.path.join(project_folder, current_datetime)
  os.makedirs(run_folder, exist_ok=True)
  os.makedirs(f"{run_folder}/images", exist_ok=True)
  os.makedirs(f"{run_folder}/checkpoints", exist_ok=True)
  return run_folder



def train(train_config,
          model_config,
          get_models,
          get_dataloader):
  
  set_seed(42)
  dynamo_backend = "no" # "inductor"
  accelerator = Accelerator(mixed_precision="fp16", dynamo_backend=dynamo_backend)
  lr = train_config.get("lr", 1e-4)
  max_iterations = train_config.get("max_iterations", None)
  debug_interval = train_config.get("debug_interval", 50)
  save_interval = train_config.get("save_interval", 1000)
  project_folder = train_config["project_folder"]
  n_epochs = train_config["n_epochs"]
  use_blur_loss = train_config["use_blur_loss"]
  start_size = model_config["start_size"]
  upscaler, task_heads = get_models()
  optimizer = torch.optim.AdamW(
      [{"params": upscaler.parameters(), "lr": lr},
      {"params": task_heads.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.0)

  dataloader = get_dataloader()
  run_folder = init_run(project_folder)
  dataloader, upscaler, task_heads, optimizer = accelerator.prepare(dataloader, upscaler, task_heads, optimizer)

  running_loss = None

  n = 0
  try:
    for epoch in range(n_epochs):
        print("Epoch", epoch + 1)
        for images, diffuse_illuminations, diffuse_reflectances, residuals, normal_bump_cams in dataloader:
            batch = dict(
                images=images,
                diffuse_illuminations=diffuse_illuminations,
                diffuse_reflectances=diffuse_reflectances,
                residuals=residuals,
                normal_bump_cams=normal_bump_cams,
            )
            if n == max_iterations:
              return
            n += 1

            with accelerator.autocast():
              out = upscaler(batch["images"], return_all_scales=True, return_original_features=True)
              results = out["all_scales"]
              features = out["original_features"]
              loss = ablation_loss([(task_heads(r), r) for r in results], batch, features, start_size, use_blur_loss)
                          
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if running_loss is None:
              running_loss = loss.item()
            else:
              running_loss = 0.99 * running_loss + 0.01 * loss.item()

            if n % debug_interval == 0 and accelerator.is_local_main_process:
              debug_step(run_folder, batch, features, results, running_loss, n)
            
            if n % save_interval == 0:
              accelerator.save_state( f"{run_folder}/checkpoints/{n}")

            del batch
            del loss
            del results
            del images
            del features
            del normal_bump_cams
            del residuals
            del diffuse_illuminations
            del diffuse_reflectances
            

  except:
    del batch
    del loss
    del optimizer
    del upscaler
    del results
    raise


 


def normalize_depths(x):
  x = x.clamp(1e-2).log()
  x = (x - x.min()) / (x.max() - x.min())
  return x
  
def debug_step_depth(run_folder, batch, predicted, features, running_loss, n):
  with torch.no_grad():
    img = torchvision.transforms.functional.to_pil_image(batch["images"][0]).resize((128, 128))
    depths = torchvision.transforms.functional.to_pil_image(normalize_depths(batch["depths"][0].repeat(3, 1, 1))).resize((128, 128))
    predicted = torchvision.transforms.functional.to_pil_image(normalize_depths(predicted[0].repeat(3, 1, 1))).resize((128, 128))
    features = debug_features(features[0], [])
    debug_image = Image.fromarray(np.hstack([img, depths, predicted, features]))
    debug_image.save(f"{run_folder}/images/{n}.jpg")
    line = f"{n} {running_loss}"
    print(line)
    with open(f"{run_folder}/training_loss.txt", "a+") as f:
      f.write(line + "\n")
    try:
      from IPython.display import display
      display(debug_image)
    except:
      pass

from safetensors import safe_open

def depth_loss(pred, gt, mask):
  loss = 0.0
  for i in range(3):
    loss += si_log_loss(pred, gt, mask)
    pred = torch.nn.functional.interpolate(pred, scale_factor=0.5, mode="bilinear")
    gt = torch.nn.functional.interpolate(gt, scale_factor=0.5, mode="bilinear")
    mask = torch.nn.functional.interpolate(mask.to(dtype=pred.dtype), scale_factor=0.5, mode="nearest") > 0
  return loss

def train_depth(train_config,
          model_config,
          get_model,
          get_dataloaders):
  
  set_seed(42)
  dynamo_backend = "no" # "inductor"
  accelerator = Accelerator(mixed_precision="fp16", dynamo_backend=dynamo_backend)

  lr = train_config.get("lr", 1e-3)
  max_iterations = train_config.get("max_iterations", None)
  debug_interval = train_config.get("debug_interval", 50)
  save_interval = train_config.get("save_interval", 1000)
  project_folder = train_config["project_folder"]
  n_epochs = train_config["n_epochs"]
  use_blur_loss = train_config["use_blur_loss"]
  n_features = train_config["n_features"]
  start_size = model_config["start_size"]
  checkpoint = train_config["checkpoint"]

  upscaler, _ = get_model()
  upscaler.eval()
  upscaler.requires_grad_ = False
  upscaler_state_dict = {}
  with safe_open(checkpoint, framework="pt", device="cpu") as f:
    for key in f.keys():
        upscaler_state_dict[key] = f.get_tensor(key)
  upscaler.load_state_dict(upscaler_state_dict)

  head = MyUNet(model_config["n_features_in"], n_features)

  import bitsandbytes 
  optimizer = bitsandbytes.optim.AdamW8bit(head.parameters(), lr=lr, eps=1e-5, weight_decay=1e-5)

  train_dataloader, val_dataloader = get_dataloaders()
  run_folder = init_run(project_folder)
  train_dataloader, val_dataloader, head, upscaler, optimizer = accelerator.prepare(train_dataloader, val_dataloader, head, upscaler, optimizer)

  running_loss = None

  n = 0
  try:
    for epoch in range(n_epochs):
        if accelerator.is_local_main_process:
          print("Epoch", epoch + 1)
        for images, depths, masks in train_dataloader:
            batch = dict(
                images=images,
                depths=depths,
            )
            if n == max_iterations:
              return
            n += 1

            with accelerator.autocast():
              with torch.no_grad():
                features = upscaler(batch["images"])
              predicted = head(features)
              loss = depth_loss(predicted, depths, masks)
                          
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if running_loss is None:
              running_loss = loss.item()
            else:
              running_loss = 0.99 * running_loss + 0.01 * loss.item()

            if n % debug_interval == 0 and accelerator.is_local_main_process:
              debug_step_depth(run_folder, batch, predicted, features, running_loss, n)
            
            if n % save_interval == 0:
              accelerator.save_state( f"{run_folder}/checkpoints/{n}")

            del batch
            del loss
            del features
            del predicted
            del images
            del depths
        with torch.no_grad():
          if accelerator.is_local_main_process:
            print(f"Epoch {epoch} done, computing validation loss")
          val_loss = 0.0
          k = 0
          for images, depths, masks in val_dataloader:
            with accelerator.autocast():
              features = upscaler(images)
              predicted = head(features)
              val_loss += depth_loss(predicted, depths, masks).item()
              k += 1
              del features
              del predicted
              del images
              del depths
              del masks
          val_loss /= k
          if accelerator.is_local_main_process:
            line = f"Epoch {epoch} validation loss: {val_loss}"
            print(line)
            with open(f"{run_folder}/validation_loss.txt", "a+") as f:
              f.write(line + "\n")


  except:
    del batch
    del loss
    del optimizer
    del upscaler
    raise