from transformers import AutoModelForDepthEstimation
import torch
import os
from datetime import datetime
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

depth_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ]).reshape((1, 3, 1, 1))

depth_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ]).reshape((1, 3, 1, 1))

depth_size = 518

def preprocess_image_for_depth(image_tensor):
  image_tensor = (image_tensor - depth_image_mean.to(device=image_tensor.device)) / depth_image_std.to(device=image_tensor.device)
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(depth_size, depth_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  return image_tensor

DA_FOCAL = 470.4

@torch.no_grad()
def get_depth_anything_depth(depth_anything, images, normalize=False):
  size = images.shape[-1]
  with torch.no_grad():
    inputs = preprocess_image_for_depth(images)
    outputs = depth_anything(pixel_values=inputs)
    predicted_depth = outputs.predicted_depth
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )
    return predicted_depth / DA_FOCAL

# @torch.compile
def mean_relative_error(input, target, mask):
  if mask is not None:
    input = input[mask]
    target = target[mask]
  # input = rescale(input)
  # target = rescale(target)
  return torch.mean(torch.abs(input - target) / (target + 1e-5)).clip(0, 10)


# @torch.compile
def si_log_loss(pred, target, mask=None):
  if mask is not None:
    mask = mask & (target > 0) & (pred > 0)
  else:
    mask = (target > 0) & (pred > 0)
  diff_log = torch.log(target[mask]) - torch.log(pred[mask])
  loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          0.5 * torch.pow(diff_log.mean(), 2))
  return loss

@torch.compile
def abs_loss(x, y, mask):
  if mask is not None:
    x = x[mask]
    y = y[mask]
  return torch.mean(torch.abs(x - y))


def init_run(project_folder):
  current_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
  run_folder = os.path.join(project_folder, current_datetime)
  os.makedirs(run_folder, exist_ok=True)
  os.makedirs(f"{run_folder}/images", exist_ok=True)
  os.makedirs(f"{run_folder}/checkpoints", exist_ok=True)
  return run_folder

from PIL import Image

def abs_depth_to_image(batch, predicted, display_size):
  depths = batch["depths"][0]
  predicted = predicted[0]
  if "masks" in batch and batch["masks"] is not None:
      masks = batch["masks"][0]
      M = max(torch.max(predicted).item(), torch.max(depths[masks]).item())
      m = min(torch.min(predicted).item(), torch.min(depths[masks]).item())
  else:
      M = max(torch.max(predicted).item(), torch.max(depths).item())
      m = min(torch.min(predicted).item(), torch.min(depths).item())
  predicted = (predicted - m) / (M - m)
  depths = (depths - m) / (M - m)
  if "masks" in batch and batch["masks"] is not None:
    depths[~masks] = 0.0
  predicted = predicted.squeeze().detach().cpu().numpy() * 255.0
  depths = depths.squeeze().detach().cpu().numpy() * 255.0
  predicted = Image.fromarray(predicted.astype(np.uint8)).resize((display_size, display_size)).convert("RGB")
  depths = Image.fromarray(depths.astype(np.uint8)).resize((display_size, display_size)).convert("RGB")
  return Image.fromarray(np.hstack([depths, predicted]).astype(np.uint8))


import numpy as np

def debug_step(run_folder, batch, results, running_loss, running_mre, n, display_size):
  with torch.no_grad():
    images = [torchvision.transforms.functional.to_pil_image(batch["images"][0])]
    depths = abs_depth_to_image(batch, results, display_size)
    images.append(depths)
    debug_image = Image.fromarray(np.hstack(images).astype(np.uint8))
    debug_image.save(f"{run_folder}/images/{n}.jpg")
    line = [str(x) for x in [n, running_loss.item(), running_mre.item()]]
    line = "\t".join(line)
    print(line)
    with open(f"{run_folder}/training_loss.txt", "a+") as f:
      f.write(line + "\n")

def debug_validation(run_folder, epoch, loss):
  line = [str(x) for x in [epoch, loss]]
  line = "\t".join(line)
  print(line)
  with open(f"{run_folder}/validation_loss.txt", "a+") as f:
    f.write(line + "\n")

DA_CACHE = {}

def make_pretraining_dataloader(dataloader, depth_anything_path, local_files_only=True):
  if depth_anything_path in DA_CACHE:
    depth_anything = DA_CACHE[depth_anything_path]
  else:
    with torch.no_grad():
      print("loading depth anything")
      depth_anything = AutoModelForDepthEstimation.from_pretrained(depth_anything_path, local_files_only=local_files_only).to(device="cuda")
      depth_anything = torch.compile(depth_anything)
      DA_CACHE[depth_anything_path] = depth_anything
    
  for images in dataloader:
    images = images.to(device="cuda", non_blocking=False)
    depths = get_depth_anything_depth(depth_anything, images)
    yield dict(
      images=images,
      depths=depths,
      masks=None
    )

def get_optimizers(model, train_dataloader, val_dataloader, lr, warmup_steps, accelerator):
  import bitsandbytes
  optimizer = bitsandbytes.optim.AdamW8bit(
[      {"params": model.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.03)

  def lr_lambda(epoch):
    return min(1, epoch / warmup_steps)
  
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
  if val_dataloader is not None:
    return accelerator.prepare(model, train_dataloader, val_dataloader, optimizer, scheduler)
  model, train_dataloader, optimizer, scheduler = accelerator.prepare(model, train_dataloader, optimizer, scheduler)
  return model, train_dataloader, val_dataloader, optimizer, scheduler

def accumulate_losses(batch, predicted, accumulation, loss, running_loss, running_mre):
  with torch.no_grad():
    mre = mean_relative_error(predicted, batch["depths"], batch["masks"])
    if running_mre is None:
      running_mre =  mre.detach()
    else:
      running_mre = 0.98 * running_mre + 0.02 *   mre.detach()

    if running_loss is None:
      running_loss = accumulation * loss.detach()
    else:
      running_loss = 0.98 * running_loss + 0.02 *  accumulation * loss.detach()
    return running_loss, running_mre

import torch.nn as nn

def train_parallel(
        get_dataloaders,
        get_model,
        train_config):
  set_seed(42)
  kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
  dynamo_backend = "inductor"
  accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[kwargs], dynamo_backend=dynamo_backend)

  n_epochs = train_config.get("n_epochs", 1)
  lr = train_config.get("lr", 1e-4)
  n_epochs = train_config.get("n_epochs", 1)
  max_iterations = train_config.get("max_iterations", None)
  debug_interval = train_config.get("debug_interval", 50)
  save_interval = train_config.get("save_interval", 1000)
  display_size = train_config.get("display_size", 128)
  warmup_steps = train_config.get("warmup_steps", 100)
  project_folder = train_config["project_folder"]

  
  model = get_model(accelerator).cuda()
  train_dataloader, val_dataloader = get_dataloaders()

  run_folder = init_run(project_folder)
  model, train_dataloader, val_dataloader, optimizer, scheduler = get_optimizers(model, train_dataloader, val_dataloader, lr, warmup_steps, accelerator)
    
  running_loss = None
  running_mre = None
  n = 0

  print("Start training")
  for epoch in range(n_epochs):
    print("Epoch", epoch)
    for batch in train_dataloader:
        if n == max_iterations:
          return
        n += 1
        with accelerator.autocast():
            predicted = model(batch["images"])
            loss = (predicted[batch["masks"]] - batch["depths"][batch["masks"]]).abs().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss, running_mre = accumulate_losses(batch, predicted, 1, loss, running_loss, running_mre)
        
        if n % debug_interval == 0 and accelerator.is_local_main_process:
          debug_step(run_folder, batch, predicted, running_loss, running_mre, n, display_size)

        if n % save_interval == 0:
          accelerator.save_state( f"{run_folder}/checkpoints/{n}")

        del batch
        del loss
        del predicted

    if val_dataloader is not None and accelerator.is_local_main_process:
      with accelerator.autocast(), torch.no_grad():
        loss = 0.0
        n = 0
        for batch in val_dataloader:
          predicted = model(batch["images"])
          loss += loss_fn(predicted, batch["depths"], batch["masks"]).item()
          del predicted
          del batch
          n += 1
        loss /= n
        debug_validation(run_folder, epoch + 1, loss)
        