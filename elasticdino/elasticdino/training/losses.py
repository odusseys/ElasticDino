import torch
import kornia
import math
import torch.nn as nn


@torch.compile(dynamic=True)
def structural_loss(x, y):
  return ((x - y) ** 2).mean() + kornia.losses.ssim_loss(x, y, 11)

@torch.compile(dynamic=True)
def gradient_loss(reproduced, truth):
  B, C, W, H = reproduced.shape
  gr = kornia.filters.spatial_gradient(reproduced).reshape((B, 2 * C, W, H))
  gt = kornia.filters.spatial_gradient(truth).reshape((B, 2 * C, W, H))
  return structural_loss(gr, gt)

@torch.compile(dynamic=True)
def laplacian_loss(reproduced, truth):
  lr = kornia.filters.laplacian(reproduced, 3)
  lt = kornia.filters.laplacian(truth, 3)
  return structural_loss(lr, lt)

def reproduction_loss(reproduced, truth, resolution, order=0):
  reproduced = torch.nn.functional.interpolate(reproduced, resolution, mode="bilinear", align_corners=False, antialias=True)
  truth = torch.nn.functional.interpolate(truth, resolution, mode="bilinear", align_corners=False, antialias=True)
  loss = structural_loss(reproduced, truth)
  if order > 0:
    loss += gradient_loss(reproduced, truth)
  if order > 1:
    loss += laplacian_loss(reproduced, truth)
  return loss


@torch.compile(dynamic=True)
def segmentation_logit_loss(logits, ground_truth_probabilities):
  sd = logits.shape[-1] / 128
  window = max(3, int(sd * 2 + 1))

  logits = kornia.filters.gaussian_blur2d(logits, window, (sd,sd))
  batch_size, n_classes, height, width = logits.shape
  ground_truth_probabilities = nn.functional.interpolate(ground_truth_probabilities, (height, width), mode="bilinear", align_corners=False, antialias=True)
  logits = logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
  ground_truth_probabilities = ground_truth_probabilities.permute(0, 2, 3, 1).reshape(batch_size * height * width, n_classes)

  return nn.functional.cross_entropy(logits, ground_truth_probabilities).mean()


def max_grad(images):
  grads = kornia.filters.sobel(images).abs()
  grads = grads / grads.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
  return grads.max(1, keepdim=True)[0]

def grid_loss(displacements):
    N = displacements.shape[1]
    normalized_displacements = (displacements + 1) * N / 2  # Maps [-1, 1] to [0, N]
    loss = torch.mean(0.5 - 0.5 * torch.cos(2 * math.pi * normalized_displacements))
    return loss

def full_loss(results_list, batch, min_scale):
  rpr = 0
  dptl = 0
  segl = 0
  diffl = 0
  diffr = 0
  resl = 0
  norml = 0
  gridl = 0
  n = 0

  for results in results_list:
    gridl += 0.0 # 0.1 * grid_loss(results["field"])
    resolution = batch["images"].shape[-1]
    scale = 1.0
    while resolution >= min_scale:
      r = results["deformed_head_results"]
      n += scale

      rpr += scale * reproduction_loss(r["reproduced"], batch["images"], resolution, order=0)
      dptl += scale * 2 * reproduction_loss(r["depth"], batch["depth"], resolution, order=2)
      segl += scale * 0.2 * segmentation_logit_loss(r["segmentation"], batch["segmentation"])

      if batch["diffuse_illuminations"] is not None:
        diffl += scale * reproduction_loss(r["diffuse_illuminations"], batch["diffuse_illuminations"], resolution, order=2)
        diffr += scale * reproduction_loss(r["diffuse_reflectances"], batch["diffuse_reflectances"], resolution, order=2)
        resl += scale * reproduction_loss(r["residuals"], batch["residuals"], resolution, order=2)
        norml += scale * reproduction_loss(r["normal_bump_cams"], batch["normal_bump_cams"], resolution, order=2)

      del r
      resolution //= 2
      scale *= 2

    del results

  gridl /= len(results_list)
  rpr = rpr / n
  dptl = dptl / n
  segl = segl / n
  diffl = diffl / n
  diffr = diffr / n
  resl = resl / n
  norml = norml / n


  res = dict(
      reproduction=rpr,
      depth_loss=dptl,
      segmentation_loss=segl,
      diffl=diffl,
      diffr=diffr,
      resl=resl,
      norml=norml,
      gridl=gridl,
      loss=(rpr + dptl + segl + diffl + diffr + resl + norml + gridl) / 7,
  )
  del rpr
  del dptl
  del segl
  del diffl
  del diffr
  del resl
  del norml
  del gridl
  return res