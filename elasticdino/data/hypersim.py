import os
import random
from PIL import Image

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



