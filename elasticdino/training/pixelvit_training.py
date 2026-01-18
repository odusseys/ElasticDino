# Doc: Adjusted from the "pixelvit.ipynb" elasticdino training python notebook.
# downloaded hypersim dataset using cmd command "gdown https://drive.google.com/uc?id=1mSrlZu1bxB-N-dHt4e4hu3_X82HEvKIp" (installed gdown with pip)

#################################################
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
#################################################

# START

#################################################
# %load_ext tensorboard # !!!!!!!!!!! perhaps need to find replacement
# from google.colab import drive
import os
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import kornia
import random
import cProfile
import pstats
import io
import math
from IPython.display import display
import gc
import torchvision.ops
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F
import cv2
# from google.colab import userdata
import os
import shutil
import requests
import zipfile
from pycocotools.coco import COCO



# wandb_key = userdata.get('wandb_login')

# !wandb login {wandb_key}

# Doc: WANDB_API_KEY env variable used implicitly

import wandb


def init_wandb_run(run_type, slug,):
  run = wandb.init(
    project="feature-upscaling",
    config={
        "run_type": run_type,
        "slug": slug,
    }
  )
  return run

def clear_cuda():
  with torch.no_grad():
    gc.collect()
    torch.cuda.empty_cache()

#################################################

#################################################
from datasets import  VerificationMode

# Doc: HF_TOKEN env variable used implicitly

imagenet_data_files = [
    f"imagenet22k-train-{i:04}.tar" for i in range(50)
]
imagenet = load_dataset("timm/imagenet-22k-wds",
                        split="train",
                        data_files=imagenet_data_files,
                        verification_mode=VerificationMode.NO_CHECKS,
                        num_proc=16)

from torch.utils.data import DataLoader

def collate_function(x):
  def process(img):
    img = img.convert("RGB")
    l = min(img.width, img.height)
    img = img.crop((0, 0, l, l))
    img = img.resize((256, 256))
    return img
  x = [process(t["jpg"]) for t in x]
  return dict(images=x)

def load_imagenet(batch_size):
  dataloader = DataLoader(imagenet, batch_size=batch_size, collate_fn=collate_function, shuffle=True)
  for x in dataloader:
    yield x
#################################################

#################################################
def list_hypersim_images(path):
  res = []
  for folder in os.listdir(f"{path}"):
    try:
      images_folders = os.listdir(f"{path}/{folder}/images")
      if "final" in images_folders[0]:
        final_folder = images_folders[0]
        geometry_folder = images_folders[1]
      else:
        final_folder = images_folders[1]
        geometry_folder = images_folders[0]
      final_files = os.listdir(f"{path}/{folder}/images/{final_folder}")
      frames = set(f.split(".")[1] for f in final_files)
      for frame in frames:
        res.append((folder, final_folder, geometry_folder, frame))
    except:
      continue
  # shuffle with fixed seed
  random.seed(42)
  random.shuffle(res)
  return res


def load_hypersim_images(path):
  for folder, final_folder, geometry_folder, frame in list_hypersim_images(path):
    try:
      image = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.color.jpg").convert("RGB")
      diffuse_illumination = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.diffuse_illumination.jpg").convert("RGB")
      diffuse_reflectance = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.diffuse_reflectance.jpg").convert("RGB")
      residual = Image.open(f"{path}/{folder}/images/{final_folder}/frame.{frame}.residual.jpg").convert("RGB")
      # semantic = Image.open(f"{path}/{folder}/images/{geometry_folder}/frame.{frame}.semantic.png")
      normal_bump_cam = Image.open(f"{path}/{folder}/images/{geometry_folder}/frame.{frame}.normal_bump_cam.png").convert("RGB")
    except:
      continue
    yield image, diffuse_illumination, diffuse_reflectance, residual, normal_bump_cam

def load_hypersim(batch_size, path="hypersim"):
  images = []
  diffuse_illuminations = []
  diffuse_reflectances = []
  residuals = []
  normal_bump_cams = []
  n = 0
  for image, diffuse_illumination, diffuse_reflectance, residual, normal_bump_cam in load_hypersim_images(path):
    images.append(image)
    diffuse_illuminations.append(diffuse_illumination)
    diffuse_reflectances.append(diffuse_reflectance)
    residuals.append(residual)
    normal_bump_cams.append(normal_bump_cam)
    n += 1
    if n == batch_size:
      yield dict(
          images=images,
          diffuse_illuminations=diffuse_illuminations,
          diffuse_reflectances=diffuse_reflectances,
          residuals=residuals,
          normal_bump_cams=normal_bump_cams
      )
      images = []
      diffuse_illuminations = []
      diffuse_reflectances = []
      residuals = []
      normal_bump_cams = []
      n = 0

#################################################

#################################################
dino_model = 's'


features_sizes = {
    's': 384,
    'b': 768,
    'l': 1024,
    'g': 1536
}

N_FEATURES = features_sizes[dino_model]


import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc

def clear_cuda():
  with torch.no_grad():
    gc.collect()
    torch.cuda.empty_cache()

MINIMAL_IMAGE_SIZE = 224
MINIMAL_GRID_SIZE = MINIMAL_IMAGE_SIZE // 14

# monkeypatch utility for striding
def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        new_w = int(math.sqrt(x.shape[1])) * self.patch_size
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, new_w, new_w)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x
import types




class DinoV2:
  def __init__(self, stride=None):
    dino_backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{dino_model}14_reg').to("cuda", torch.float32)
    dino_backbone.eval()
    if stride is not None:
      dino_backbone.patch_embed.proj.stride = (stride, stride)
      dino_backbone.prepare_tokens_with_masks = types.MethodType(prepare_tokens_with_masks, dino_backbone)
    dino_backbone.requires_grad_(False)
    self.dino_backbone = torch.compile(dino_backbone)
    pass

  def prepare_images(self, images, image_check=True):
    tensors = [transforms.functional.pil_to_tensor(i) for i in images]
    tensors = torch.stack(tensors)
    tensors = tensors.to(dtype=torch.float32, device="cuda") / 255.0
    return tensors

  # Doc: changed code to include normalization
  # def get_features_for_tensor(self, images, image_check=True):
  #   with torch.no_grad():
  #     res = self.dino_backbone.get_intermediate_layers(images, n=1)[-1]
  #     grid_size = int(math.sqrt(res.shape[1]))
  #     res = res.reshape((res.shape[0], grid_size, grid_size, N_FEATURES)).permute((0, 3, 1, 2))
  #     del images
  #     return res

  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Version with normaliation (the transform)

  def get_features_for_tensor(self, images, image_check=True):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    with torch.no_grad():
      images = transform(images)
      res = self.dino_backbone.get_intermediate_layers(images, n=1)[-1]
      grid_size = int(math.sqrt(res.shape[1]))
      res = res.reshape((res.shape[0], grid_size, grid_size, N_FEATURES)).permute((0, 3, 1, 2))
      del images
      return res

  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  def get_features(self, images, image_check=True, one_by_one=True):
    images = self.prepare_images(images, image_check)
    return self.get_features_for_tensor(images, image_check)
#################################################

#################################################
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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
#################################################

#################################################
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# load Mask2Former fine-tuned on COCO panoptic segmentation
segmentation_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic").cuda()
# segmentation_model = torch.compile(segmentation_model, mode="reduce-overhead", dynamic=True)

def make_segmentation_probabilities(outputs):
  class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
  masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

  # Scale back to preprocessed image size - (384, 384) for all models
  masks_queries_logits = torch.nn.functional.interpolate(
      masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
  )

  # Remove the null class `[..., :-1]`
  masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
  masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

  # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
  segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
  return segmentation

segmentation_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ], device="cuda").reshape((1, 3, 1, 1))

segmentation_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ], device="cuda").reshape((1, 3, 1, 1))

segmentation_image_size = 384

def preprocess_image_for_segmentation(image_tensor):
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(segmentation_image_size, segmentation_image_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  image_tensor = (image_tensor - segmentation_image_mean) / segmentation_image_std

  return image_tensor

def get_segmentation(images):
  size = images.shape[-1]
  with torch.no_grad():
    inputs = preprocess_image_for_segmentation(images)
    outputs = segmentation_model(pixel_values=inputs)
    probas = make_segmentation_probabilities(outputs)
    return torch.nn.functional.interpolate(
        probas,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )



def get_segmentation_map(image, mask):
  image = (image * 255).to(dtype=torch.uint8)

  # make boolean segmentation masks
  mask = torch.nn.functional.interpolate(
      mask.unsqueeze(0),
      size=image.shape[-1],
      mode="nearest",
  ).squeeze(1).to(dtype=torch.long)
  one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=N_SEGMENTATION_CLASSES + 1).squeeze()
  one_hot_mask = one_hot_mask.permute(2, 0, 1).contiguous()
  boolean_mask = one_hot_mask.type(torch.bool)
  res = torchvision.utils.draw_segmentation_masks(image, boolean_mask).permute(1, 2, 0)
  return Image.fromarray(res.cpu().numpy())
#################################################

#################################################
class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def fit(self, X):
        b, n, d = X.shape
        self.register_buffer("mean_", X.mean(1, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh.transpose(1, 2)[:, :, :self.n_components]
        self.register_buffer("components_", Vt)
        std = S[:, :self.n_components].unsqueeze(1).sqrt()
        self.register_buffer("std_", std)
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        unscaled = torch.bmm(X - self.mean_, self.components_)
        scaled = unscaled / self.std_  # Scale for unit variance
        return scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        Y = Y * self.std_  # Unscale
        return  torch.bmm(Y, self.components_.transpose(1, 2)) + self.mean_


@torch.compile
def compute_pca(f1, n):
  f1_size = f1.shape[-1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, N_FEATURES, f1_size * f1_size)).transpose(1, 2)
  pca = PCA(n)
  f1_reduced = pca.fit_transform(f1)
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  return f1_reduced, pca

@torch.compile
def apply_pca(pca, f1):
  f1_size = f1.shape[-1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, N_FEATURES, f1_size * f1_size)).transpose(1, 2)
  f1_reduced = pca.transform(f1)
  n = f1_reduced.shape[2]
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  return f1_reduced

def reduce_features(f1, f2, n):
  f1_size = f1.shape[-1]
  f2_size = f2.shape[-1]
  batch_size = f1.shape[0]
  f1 = f1.reshape((batch_size, N_FEATURES, f1_size * f1_size)).transpose(1, 2)
  f2 = f2.reshape((batch_size, N_FEATURES, f2_size * f2_size)).transpose(1, 2)
  pca = PCA(n)
  f1_reduced = pca.fit_transform(f1)
  f2_reduced = pca.transform(f2)
  f1_reduced = f1_reduced.transpose(1, 2).reshape((batch_size, n, f1_size, f1_size))
  f2_reduced = f2_reduced.transpose(1, 2).reshape((batch_size, n, f2_size, f2_size))
  return f1_reduced, f2_reduced, pca

def reduce_dimension(f1, other_features, n_features, n):
  batch_size = f1.shape[0]
  size_1 = f1.shape[2]
  other_sizes = [f2.shape[2] for f2 in other_features]
  f1 = f1.permute((0, 2, 3, 1)).reshape((batch_size, size_1 * size_1, n_features))
  other_features = [f2.permute((0, 2, 3, 1)).reshape((batch_size, f2.shape[2] * f2.shape[2], n_features)) for f2 in other_features]
  pca = PCA(n_components=n).fit(f1)
  f1 = pca.transform(f1).reshape((batch_size, size_1, size_1, n)).permute(0, 3, 1, 2)
  other_features = [pca.transform(f2).reshape((batch_size, size_2, size_2, n)).permute(0, 3, 1, 2) for f2, size_2 in zip(other_features, other_sizes)]
  m = min(torch.min(f1), *[torch.min(f2) for f2 in other_features])
  M = max(torch.max(f1), *[torch.max(f2) for f2 in other_features])
  f1 = (f1 - m) / (M - m)
  other_features = [(f2 - m) / (M - m) for f2 in other_features]
  return f1, other_features


def random_projection(features_list, n_projected_features, dtype=torch.float32):
  proj = torch.randn(features_list[0].shape[0],
                    features_list[0].shape[1],
                    n_projected_features, device="cuda", dtype=dtype, requires_grad=False)
  proj /= proj.square().sum(1, keepdim=True).sqrt()
  return [torch.einsum("bchw,bcd->bdhw", features, proj) for features in features_list]


def debug_features(f1, other_features, display_size=128):
  f1 = f1.to(dtype=torch.float32)
  other_features = [f2.to(dtype=torch.float32) for f2 in other_features]
  n_features = f1.shape[0]
  f1, other_features = reduce_dimension(f1.unsqueeze(0), [f2.unsqueeze(0) for f2 in other_features], n_features, 3)
  f1 = f1[0].permute(1, 2, 0).detach().cpu().float().numpy().squeeze() * 255
  images = [Image.fromarray(f1.astype(np.uint8)).resize((display_size, display_size), 0)]
  for f2 in other_features:
    f2 = f2[0].permute(1, 2, 0).detach().cpu().float().numpy().squeeze() * 255
    images.append(Image.fromarray(f2.astype(np.uint8)).resize((display_size, display_size), 0))
  return Image.fromarray(np.hstack(images).astype(np.uint8))

def debug_individual_features(features, size=128, clip=5):
  images = []
  for i in range(min(5, len(features))):
    f = np.clip(features[i], -clip, clip)
    f = (f + clip) / (2 * clip) * 255
    images.append(Image.fromarray(f.astype(np.uint8)).resize((size, size), 0))
  # make row of images and display
  display(Image.fromarray(np.hstack(images).astype(np.uint8)))
#################################################

#################################################
Activation = nn.GELU

NormLayer = nn.BatchNorm2d

class DepthwiseConvolution(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, padding_mode="replicate", **kwargs):
        super(DepthwiseConvolution, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, groups=nin, **kwargs)
        self.pointwise = nn.Conv2d(nin, nout,kernel_size=1, padding=0,   **kwargs )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ResidualBlock(nn.Module):
  def __init__(self, out_channels, kernel_size=3, padding=1, depthwise=True, dtype=torch.float32,
               padding_mode="replicate", shrinkage=1.0):
    super().__init__()
    Layer = DepthwiseConvolution if depthwise else nn.Conv2d
    self.layers = nn.Sequential(
        Layer(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dtype=dtype, bias=False),
        NormLayer(out_channels, dtype=dtype),
        Activation(),
        Layer(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dtype=dtype, bias=False),
        NormLayer(out_channels, dtype=dtype),
        Activation(),
    )
    self.shrinkage = shrinkage

  def forward(self, x):
    l = self.layers(x)
    return x + self.shrinkage * l

class ProjectionLayer(nn.Module):
  def __init__(self, n_features_in, n_features_out):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(n_features_in, n_features_out, 1, bias=False),
        NormLayer(n_features_out),
    )

  def forward(self, x):
    return self.layers(x)

class FCLayer(nn.Module):
  def __init__(self,  n_features_in, n_features_out=None, residual=False, bn=True, shrinkage=1.0):
    super().__init__()
    if n_features_out is None:
      n_features_out = n_features_in
    self.residual = residual if n_features_in == n_features_out else False
    self.layers = nn.Sequential(
        nn.Conv2d(n_features_in, n_features_out, 1, bias=not bn),
        Activation(),
    )
    self.shrinkage = shrinkage
    if bn:
      self.bn = NormLayer(n_features_out)
    else:
      self.bn = None

  def forward(self, f):
    res = self.layers(f)
    if self.residual:
      return f + self.shrinkage * res
    if self.bn is None:
      return res
    return self.bn(res)
#################################################

#################################################
import rff

import torch
import torch.nn.functional as F

dino = DinoV2()

def make_batch_data(batch, starting_size, target_size, dtype):

    def to_tensor(images):
        return torch.stack([transforms.PILToTensor()(img) for img in images]).to(device="cuda", dtype=dtype) / 255.0

    def target_resize(x):
        return nn.functional.interpolate(x, (target_size, target_size), mode="bilinear", align_corners=False, antialias=True)


    originals = [u.resize((512, 512)) for u in batch["images"]]
    images = to_tensor(originals)

    factor = starting_size // MINIMAL_GRID_SIZE

    images_small = torch.nn.functional.interpolate(images, factor * MINIMAL_IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True)
    # add small noise to remove positional encoding artifacts
    # images_small = (images_small + torch.randn_like(images_small) * 0.05).clamp(0, 1)
    features = dino.get_features_for_tensor(images_small.to(dtype=torch.float32)).to(dtype=dtype)

    images = target_resize(images)
    segmentation = get_segmentation(images).to(dtype=dtype)
    depth = get_depth(images).to(dtype=dtype)

    if "diffuse_illuminations" in batch:
      diffuse_illuminations = target_resize(to_tensor(batch["diffuse_illuminations"]))
    else:
      diffuse_illuminations = None

    if "diffuse_reflectances" in batch:
      diffuse_reflectances = target_resize(to_tensor(batch["diffuse_reflectances"]))
    else:
      diffuse_reflectances = None

    if "residuals" in batch:
      residuals = target_resize(to_tensor(batch["residuals"]))
    else:
      residuals = None

    if "normal_bump_cams" in batch:
      normal_bump_cams = target_resize(to_tensor(batch["normal_bump_cams"]))
    else:
      normal_bump_cams = None


    res = dict(features=features,
               images=images,
               depth=depth,
               segmentation=segmentation,
               diffuse_illuminations=diffuse_illuminations,
               diffuse_reflectances=diffuse_reflectances,
               residuals=residuals,
               normal_bump_cams=normal_bump_cams,
               originals=originals,
               )
    del features
    del images_small
    del images
    del depth
    del segmentation
    del diffuse_illuminations
    del diffuse_reflectances
    del residuals
    del normal_bump_cams

    return res

def get_mixed_data(batch_size, starting_size, target_size, hypersim=True, imagenet=True, dtype=torch.float32):
  hypersim_iterator = load_hypersim(batch_size)
  imagenet_iterator = load_imagenet(batch_size)

  n = 0
  while True:
    n += 1
    if n % 2 == 0:
      if not hypersim:
        continue
      x = next(hypersim_iterator, None)
      if x is None:
        hypersim_iterator = load_hypersim(batch_size)
        x = next(hypersim_iterator, None)
    else:
      if not imagenet:
        continue
      x = next(imagenet_iterator, None)
      if x is None:
        imagenet_iterator = load_imagenet(batch_size)
        x = next(imagenet_iterator, None)

    yield make_batch_data(x, starting_size, target_size, dtype)


def make_base_locations(batch_size, size, dtype):
    x = torch.arange(size, device="cuda", dtype=dtype) * (2 / size) - 1
    y = torch.arange(size, device="cuda", dtype=dtype) * (2 / size) - 1
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    res = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return res

def display_array_image(x, size=128):
  x = 255 * (x - torch.min(x)) / (torch.max(x) - torch.min(x))
  x = x.permute((1, 2, 0)).squeeze().detach().cpu().float().numpy()
  display(Image.fromarray(x.astype(np.uint8)).resize((size, size)))


@torch.compile(dynamic=True)
def structural_loss(x, y):
  return ((x - y) ** 2).mean() + kornia.losses.ssim_loss(x, y, 11)
#################################################

# MODEL - LOSSES

#################################################
class LossTracker:
  def __init__(self, names, decay=0.99):
    self.names = names
    self.losses = None
    self.decay = decay

  def track(self, losses):
    if self.losses is None:
      self.losses = [x if type(x) == float else x.detach() for x in losses]
    else:
      for i in range(len(losses)):
        l = losses[i]
        loss = l if type(l) == float else l.detach()
        self.losses[i] = self.losses[i] * self.decay + loss * (1 - self.decay)
        del l

  def debug(self):
    for i in range(len(self.names)):
      print(self.names[i], ": ", self.losses[i])


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
#################################################

#################################################
from re import X
N_SEGMENTATION_CLASSES = 133

class DeformerBlock(nn.Module):
  def __init__(self, n_layers, n_features, n_features_in):
    super().__init__()

    self.image_encoder = ProjectionLayer(3, n_features)

    self.feature_encoder = ProjectionLayer(n_features_in, n_features)

    self.convs = nn.Sequential(
        ProjectionLayer(n_features * 2, n_features),
        *[torch.compile(ResidualBlock(n_features), dynamic=True) for _ in range(n_layers)]
    )

    last_layer = nn.Conv2d(n_features // 8, 2, 1)

    self.deformer = torch.compile(nn.Sequential(
        nn.Conv2d(n_features, n_features // 2, 1),
        Activation(),
        nn.Conv2d(n_features // 2, n_features // 4, 1),
        Activation(),
        nn.Conv2d(n_features // 4, n_features // 8, 1),
        Activation(),
        last_layer,
    ), dynamic=True)

    # initialize all deformer weights to 0
    torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.003, generator=None)
    nn.init.zeros_(last_layer.bias)



  def forward(self, features, image):
    f = self.feature_encoder(features)
    image = self.image_encoder(image)
    f = self.convs(torch.cat([f, image], dim=1))
    base_locations = make_base_locations(image.shape[0], image.shape[-1], dtype=image.dtype).permute((0, 3, 1, 2))
    field = base_locations + self.deformer(f)
    field = field.permute((0, 2, 3, 1))
    return torch.nn.functional.grid_sample(features, field, padding_mode="border", align_corners=False)

class PixelViTStage(nn.Module):
  def __init__(self, layer_config, n_features_in):
    super().__init__()
    self.blocks = nn.ModuleList([
        DeformerBlock(layer_config["layers_per_block"], layer_config["hidden_features"], n_features_in)
        for _ in range(layer_config["n_blocks"])
    ])

  def forward(self, features, images):
    images = torch.nn.functional.interpolate(images, features.shape[-1], mode="bilinear")
    for block in self.blocks:
      features = block(features, images)
    return dict(deformed=features)

class PixelViT(nn.Module):
  def __init__(self, config):
    super().__init__()

    n_features_in = config["n_features_in"]
    layer_configs = config["layers"]

    n_upscales = int(math.log2(config["target_size"] // config["start_size"])) + 1
    assert n_upscales == len(layer_configs), "Incompatible resolutions and feature config"
    self.n_features_in = n_features_in
    self.stages = nn.ModuleList([
        PixelViTStage(layer_configs[res], n_features_in) for res in layer_configs
    ])

  def forward(self, features, images):
    outputs = []
    n = len(self.stages)
    current_size = features.shape[-1]
    for i in range(n):
      res = self.stages[i](features, images)
      outputs.append(res)
      if i < n - 1:
        current_size *= 2
        features = torch.nn.functional.interpolate(res["deformed"], current_size, mode="nearest")
      del res
    return outputs


class TaskHeads(nn.Module):
  def __init__(self, n_features_in):
    super().__init__()
    def make_head(dim):
      return nn.Sequential(
        nn.Conv2d(n_features_in, n_features_in // 2, 1),
        nn.GELU(),
        nn.Conv2d(n_features_in // 2, n_features_in // 4, 1),
        nn.GELU(),
        nn.Conv2d(n_features_in // 4, n_features_in // 8, 1),
        nn.GELU(),
        nn.Conv2d(n_features_in // 8, dim, 1),
    )

    self.reproduction_head = make_head(3)

    self.depth_head = make_head(1)

    self.segmentation_head = nn.Sequential(
        FCLayer(n_features_in, n_features_in),
        FCLayer(n_features_in, n_features_in),
        nn.Conv2d(n_features_in, N_SEGMENTATION_CLASSES, 1),
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
#################################################

#################################################
import lightning as L

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class MyModule(L.LightningModule):
    def __init__(self,
                  config,
                  lr = 1e-2,
                  batch_size = 4,
                  accumulation=1,
                  max_iterations=None,
                  debug_interval=51,
                  use_wandb=False,
                  display_size=128,
                  decay_period=8000,
                  save_interval=5000,
                  checkpoint=None):
        super().__init__()
        self.config = config
        self.lr = lr
        self.batch_size = batch_size
        self.accumulation = accumulation
        self.max_iterations = max_iterations
        self.debug_interval = debug_interval
        self.use_wandb = use_wandb
        self.display_size = display_size
        self.decay_period = decay_period
        self.save_interval = save_interval
        self.checkpoint = checkpoint
        self.start_size = config["start_size"]
        self.target_size = config["target_size"]

        self.upscaler = PixelViT(config)
        self.task_heads = TaskHeads(N_FEATURES)

        if checkpoint is not None:
          self.upscaler.load_state_dict(torch.load(f"upscaler-{checkpoint}"))
          self.task_heads.load_state_dict(torch.load(f"task_heads-{checkpoint}"))

    def training_step(self, batch, batch_idx):
        results = self.upscaler(batch["features"].to(memory_format=torch.channels_last), batch["images"].to(memory_format=torch.channels_last))
        for r in results:
          r["deformed_head_results"] = self.task_heads(r["deformed"])

        losses = full_loss(results, batch, self.start_size)
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = bitsandbytes.optim.AdamW8bit(
          [{"params": self.upscaler.parameters(), "lr": self.lr},
          {"params": self.task_heads.parameters(), "lr": self.lr}], eps=1e-5, weight_decay=0.0)
        if checkpoint is not None:
          optimizer.load_state_dict(torch.load(f"optimizer-{checkpoint}"))
        return optimizer
#################################################

# UPSCALER

#################################################
clear_cuda()


def debug_step(batch, results, running_loss, n, display_size, loss_tracker, use_wandb):
  with torch.no_grad():
    def make_reproduced(x):
      x = (torch.clamp(x[0], 0, 1).permute((1, 2, 0)) * 255.0).detach().cpu().float().numpy().astype(np.uint8)
      return Image.fromarray(x).resize((display_size, display_size))

    reproduced = [batch["originals"][0].resize((display_size, display_size)),
                  *[make_reproduced(r["deformed_head_results"]["reproduced"]) for r in results]]

    reproduced = Image.fromarray(np.hstack(reproduced).astype(np.uint8))

    depth_reproduced = [batch["depth"][0],
                        *[torch.clamp(r["deformed_head_results"]["depth"][0], 0, 1) for r in results]]
    depth_reproduced = [Image.fromarray((x.squeeze() * 255.0).detach().cpu().float().numpy().astype(np.uint8)).resize((display_size, display_size)) for x in depth_reproduced]
    depth_reproduced = Image.fromarray(np.hstack(depth_reproduced).astype(np.uint8)).convert("RGB")

    def make_seg(x):
      x = torch.nn.functional.interpolate(x.unsqueeze(0), size=batch["segmentation"].shape[-1], mode="nearest").squeeze(0)
      x = torch.argmax(x, dim=0).to(dtype=torch.uint8).squeeze()
      return x

    segmentation_truth = torch.argmax(batch["segmentation"][0], dim=0).to(dtype=torch.uint8).squeeze()
    segmentation_reproduced = [segmentation_truth, *[make_seg(r["deformed_head_results"]["segmentation"][0]) for r in results]]
    del segmentation_truth
    segmentation_reproduced = [get_segmentation_map(batch["images"][0], x).resize((display_size,display_size)) for x in segmentation_reproduced]
    segmentation_reproduced = Image.fromarray(np.hstack(segmentation_reproduced).astype(np.uint8))

    feature_debug = debug_features(batch["features"][0], [r["deformed"][0] for r in results], display_size)

    debug_image = Image.fromarray(np.vstack([feature_debug, reproduced, depth_reproduced, segmentation_reproduced]).astype(np.uint8))

    if use_wandb:
      wandb.log({
          "loss": running_loss,
          "debug_image": wandb.Image(debug_image),
      })
    print("Iteration", n, ":", running_loss, )
    loss_tracker.debug()
    display(debug_image)

import bitsandbytes

import cProfile
import pstats
import io
import json

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def train(config,
                  lr = 1e-2,
                  batch_size = 4,
                  accumulation=1,
                  max_iterations=None,
                  debug_interval=51,
                  use_wandb=False,
                  display_size=128,
                  decay_period=8000,
                  save_interval=5000,
                  checkpoint=None):
  start_size = config["start_size"]
  target_size = config["target_size"]

  upscaler = PixelViT(config).to("cuda", memory_format=torch.channels_last)
  task_heads = TaskHeads(N_FEATURES).to("cuda", memory_format=torch.channels_last)

  if checkpoint is not None:
    upscaler.load_state_dict(torch.load(f"upscaler-{checkpoint}"))
    task_heads.load_state_dict(torch.load(f"task_heads-{checkpoint}"))

  upscaler.train()
  task_heads.train()

  optimizer = bitsandbytes.optim.AdamW8bit(
      [{"params": upscaler.parameters(), "lr": lr},
      {"params": task_heads.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.0)
  scaler = torch.amp.GradScaler()

  def lr_lambda(epoch):
    return math.pow(10, - epoch / decay_period)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

  if checkpoint is not None:
    optimizer.load_state_dict(torch.load(f"optimizer-{checkpoint}"))
    try:
      scaler.load_state_dict(torch.load(f"scaler-{checkpoint}"))
    except:
      scaler = torch.amp.GradScaler(init_scale=2.0 ** 6)

  if use_wandb:
    run = init_wandb_run("feature_fixer", "reproduction")
    wandb.watch(upscaler, log_freq=100)

  if use_wandb:
    with open("config.json", "w+") as f:
      json.dump(config, f)
      wandb.save("config.json")

  # OPTIMIZER CONFIG

  loss_tracker = LossTracker(["reprod", "depth", "segmentation", "diffl", "diffr", "resl", "norml", "gridl",
                               "loss"], decay=0.98)
  running_loss = None
  n = 0
  try:
    for epoch in range(5):
        for batch in get_mixed_data(batch_size, start_size, target_size):
            if n == max_iterations:
              return
            n += 1
            with torch.autocast(device_type='cuda', dtype=torch.float16), torch.set_grad_enabled(True):
              results = upscaler(batch["features"].to(memory_format=torch.channels_last), batch["images"].to(memory_format=torch.channels_last))
              for r in results:
                r["deformed_head_results"] = task_heads(r["deformed"])

              losses = full_loss(results, batch, start_size)
              loss = losses["loss"] / accumulation

            scaler.scale(loss).backward()
            scheduler.step()
            if n % accumulation == 0:
              scaler.step(optimizer)
              scaler.update()
              optimizer.zero_grad()

            if n % debug_interval == 0:
              print("effective lr", scheduler.get_lr())
              debug_step(batch, results, running_loss, n, display_size, loss_tracker, use_wandb)

            loss_tracker.track([
                losses["reproduction"],
                losses["depth_loss"],
                losses["segmentation_loss"],
                losses["diffl"],
                losses["diffr"],
                losses["resl"],
                losses["norml"],
                losses["gridl"],
                losses["loss"]])

            if running_loss is None:
              running_loss = accumulation * loss.detach()
            else:
              running_loss = 0.99 * running_loss + 0.01 *  accumulation * loss.detach()

            if n % save_interval == 0:
              torch.save(upscaler.state_dict(), f"upscaler-{epoch}-{n}.pth")
              torch.save(task_heads.state_dict(), f"task_heads-{epoch}-{n}.pth")
              torch.save(optimizer.state_dict(), f"optimizer-{epoch}-{n}.pth")
              torch.save(scaler.state_dict(), f"scaler-{epoch}-{n}.pth")
              if use_wandb:
                wandb.save(f"upscaler-{epoch}-{n}.pth")
                wandb.save(f"task_heads-{epoch}-{n}.pth")
                wandb.save(f"optimizer-{epoch}-{n}.pth")
                wandb.save(f"scaler-{epoch}-{n}.pth")


            del batch
            del loss
            del results
            del losses

  except:

    if use_wandb:
      run.finish()
    del batch
    del loss
    del optimizer
    del upscaler
    del scaler
    del results
    del loss_tracker
    del losses

    raise

import sys


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def repair_checkpoint(path):
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

checkpoint = None # "0-45000.pth"

if checkpoint is not None:
  repair_checkpoint(f"upscaler-{checkpoint}")
  repair_checkpoint(f"task_heads-{checkpoint}")
  repair_checkpoint(f"optimizer-{checkpoint}")

config = dict(
    n_features_in=1024,
    layers={
        # 32: dict(hidden_features=1024, n_blocks=5, layers_per_block=8),
        64: dict(hidden_features=512, n_blocks=3, layers_per_block=6),
        128: dict(hidden_features=256, n_blocks=2, layers_per_block=6),
        256: dict(hidden_features=128, n_blocks=1, layers_per_block=4),
    },
    start_size=64,
    target_size=256,
)

train(config,
      batch_size=2,
      accumulation=4,
      lr=3e-4,
      debug_interval=101,
      decay_period=60000,
      use_wandb=True,
      checkpoint=checkpoint)
#################################################

# DOWNSTREAM TASKS