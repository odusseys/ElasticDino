from transformers import AutoModelForDepthEstimation
import torch
import torch.nn as nn
import wandb
from elasticdino.model.elasticdino import ElasticDino
import math
import bitsandbytes

depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()
depth_anything = torch.compile(depth_anything)

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


def get_depth(images):
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

@torch.compile
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

def init_wandb_run(model, dataset):
  run = wandb.init(
    project=f"depth-{model}-{dataset}",
  )
  return run


def train_depth(config,
          dataset,
          model,
          lr = 1e-2,
          decay_period=5000,
          batch_size = 4,
          accumulation=1,
          max_iterations=None,
          debug_interval=51,
          save_interval=100,
          use_wandb=False,
          display_size=128,

          n_validation_batches=None,
          checkpoint=None):

  start_size = config["start_size"]
  target_size = config["target_size"]

  elasticdino = ElasticDino.from_config(config, True)

head = UNet().cuda()
  head = torch.compile(head)

  if use_wandb:
    run = init_wandb_run(dataset, model)
    wandb.watch(elasticdino, log_freq=100)

  optimizer = bitsandbytes.optim.AdamW8bit(
[      {"params": head.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.05)
  scaler = torch.amp.GradScaler()

  def lr_lambda(epoch):
    return math.pow(10, - epoch / decay_period)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

  running_loss = None
  running_mre = None
  n = 0
  data = get_data_nyu("train", batch_size, start_size, target_size) if dataset == "nyu" else get_data_da(batch_size, start_size, target_size)
  try:
    for epoch in range(200):
      for batch in data:
          if n == max_iterations:
            return
          if batch["images"].shape[0] != batch_size:
            continue
          n += 1
          with torch.autocast(device_type='cuda', dtype=torch.float16), torch.set_grad_enabled(True):
            with torch.no_grad():
              upscaled = elasticdino(batch["images"])[-1]["deformed"]
            predicted = head(upscaled)
            loss = loss_fn(predicted, batch["depths"], batch["masks"]) / accumulation

          scaler.scale(loss).backward()
          scheduler.step()

          if n % accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

          with torch.no_grad():
            mre = mean_relative_error(predicted, batch["depths"], batch["masks"])
            if running_mre is None:
              running_mre =  mre.detach()
            else:
              running_mre = 0.98 * running_mre + 0.02 *   mre.detach()
          if n % debug_interval == 0:
            # val_loss, val_mre = validation_loss(n_validation_batches, elasticdino, batch_size, head, start_size, target_size)
            val_loss, val_mre = 0.0, 0.0
            debug_step(batch, upscaled, predicted, running_loss, running_mre, val_loss, val_mre, n, display_size, use_wandb)

          if running_loss is None:
            running_loss = accumulation * loss.detach()
          else:
            running_loss = 0.98 * running_loss + 0.02 *  accumulation * loss.detach()

          if n % save_interval == 0:
            torch.save(head.state_dict(), f"depth-head-{epoch}-{n}.pth")
            if use_wandb:
              wandb.save(f"depth-head-{epoch}-{n}.pth")

          del batch
          del loss
          del predicted
          del upscaled




  except:
    if use_wandb:
      run.finish()
    del batch
    del loss
    del optimizer
    del elasticdino
    del scaler
    del predicted

    raise
