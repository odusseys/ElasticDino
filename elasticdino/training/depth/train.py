from transformers import AutoModelForDepthEstimation
import torch
import os
from datetime import datetime
import torchvision

try:
  import bitsandbytes
  BITS_AND_BYTES = True
except:
  BITS_AND_BYTES = False

depth_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ], device="cuda").reshape((1, 3, 1, 1))

depth_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ], device="cuda").reshape((1, 3, 1, 1))

depth_size = 518

def preprocess_image_for_depth(image_tensor):
  image_tensor = (image_tensor - depth_image_mean) / depth_image_std
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(depth_size, depth_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  return image_tensor


@torch.no_grad()
def get_depth_anything_depth(depth_anything, images):
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
    predicted_depth[torch.isnan(predicted_depth)] = 0
    m = predicted_depth.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    M = predicted_depth.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    predicted_depth = (predicted_depth - m) / (M - m + 1e-5)
    return predicted_depth

# @torch.compile
def mean_relative_error(input, target, mask):
  if mask is not None:
    input = input[mask]
    target = target[mask]
  # input = rescale(input)
  # target = rescale(target)
  return torch.mean(torch.abs(input - target) / (target + 1e-5)).clip(0, 10)



def abs_loss(x, y, mask):
  if mask is not None:
    x = x[mask]
    y = y[mask]
  return torch.mean(torch.abs(x - y))

loss_fn = abs_loss

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
    

def get_optimizers(model, lr):
  optimizer_class = bitsandbytes.optim.AdamW8bit if BITS_AND_BYTES else torch.optim.AdamW
  optimizer = optimizer_class(
[      {"params": model.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.03)
  scaler = torch.amp.GradScaler()

  def lr_lambda(epoch):
    return 1
    # return math.pow(10, - epoch / decay_period)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
  return optimizer, scaler, scheduler


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

def train_depth(
          project_folder,
          dataloader,
          model,
          n_epochs=200,
          lr = 1e-2,
          decay_period=5000,
          accumulation=1,
          max_iterations=None,
          debug_interval=51,
          save_interval=100,
          display_size=128):
  # start_size = config["start_size"]
  # target_size = config["target_size"]
  run_folder = init_run(project_folder)
  optimizer, scaler, scheduler = get_optimizers(model, lr)
  running_loss = None
  running_mre = None
  n = 0
  print("Start training")
  try:
    for epoch in range(n_epochs):
      print("Epoch", epoch)
      for batch in dataloader:
          print("batch")
          if n == max_iterations:
            return
          n += 1
          with torch.autocast(device_type='cuda', dtype=torch.float16), torch.set_grad_enabled(True):
            predicted = model(batch["images"])
            print("loss")
            loss = loss_fn(predicted, batch["depths"], batch["masks"]) / accumulation

          print("back")
          scaler.scale(loss).backward()
          scheduler.step()

          if n % accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

          running_loss, running_mre = accumulate_losses(batch, predicted, accumulation, loss, running_loss, running_mre)
          
          if n % debug_interval == 0:
            debug_step(run_folder, batch, predicted, running_loss, running_mre, n, display_size)

          if n % save_interval == 0:
            torch.save(model.state_dict(), f"{run_folder}/checkpoints/{n}.pth")


          del batch
          del loss
          del predicted


  except:
    del batch
    del loss
    del optimizer
    del scaler
    del predicted

    raise


import torch.nn as nn

def train_parallel(
        project_folder,
        dataloader,
        model,
        n_epochs=200,
        lr = 1e-4,
        decay_period=5000,
        accumulation=1,
        max_iterations=None,
        debug_interval=51,
        save_interval=100,
        display_size=128):

  print("setup")
  model = nn.DataParallel(model, device_ids=[0, 1])
  model = model.cuda()

  run_folder = init_run(project_folder)
  optimizer, scaler, scheduler = get_optimizers(model, lr)
  running_loss = None
  running_mre = None
  n = 0

  print("Start training")
  for epoch in range(n_epochs):
    print("Epoch", epoch)
    for batch in dataloader:
        if n == max_iterations:
          return
        n += 1
        with torch.autocast(device_type='cuda', dtype=torch.float16), torch.set_grad_enabled(True):
          predicted = model(batch["images"])
          loss = loss_fn(predicted, batch["depths"], batch["masks"]) / accumulation

        scaler.scale(loss).backward()
        scheduler.step()

        if n % accumulation == 0:
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()

        running_loss, running_mre = accumulate_losses(batch, predicted, accumulation, loss, running_loss, running_mre)
        
        if n % debug_interval == 0:
          debug_step(run_folder, batch, predicted, running_loss, running_mre, n, display_size)

        if n % save_interval == 0:
          torch.save(model.state_dict(), f"{run_folder}/checkpoints/{n}.pth")


        del batch
        del loss
        del predicted
