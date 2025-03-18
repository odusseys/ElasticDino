import os
import random
from PIL import Image
import torch
import torchvision
import concurrent.futures

def make_paths(base_path, folder, final_folder, geometry_folder, frame):
  image = f"{base_path}/{folder}/images/{final_folder}/frame.{frame}.color.jpg"
  diffuse_illumination = f"{base_path}/{folder}/images/{final_folder}/frame.{frame}.diffuse_illumination.jpg"
  diffuse_reflectance = f"{base_path}/{folder}/images/{final_folder}/frame.{frame}.diffuse_reflectance.jpg"
  residual = f"{base_path}/{folder}/images/{final_folder}/frame.{frame}.residual.jpg"
  normal_bump_cam = f"{base_path}/{folder}/images/{geometry_folder}/frame.{frame}.normal_bump_cam.png"
  return image, diffuse_illumination, diffuse_reflectance, residual, normal_bump_cam

def check_paths_ok(path_params):
  for path in make_paths(*path_params):
    if not os.path.isfile(path):
      return False
  return True

def list_hypersim_images(base_path):
  res = []
  for folder in os.listdir(f"{base_path}"):
    try:
      images_folders = os.listdir(f"{base_path}/{folder}/images")
      if "final" in images_folders[0]:
        final_folder = images_folders[0]
        geometry_folder = images_folders[1]
      else:
        final_folder = images_folders[1]
        geometry_folder = images_folders[0]
      final_files = os.listdir(f"{base_path}/{folder}/images/{final_folder}")
      frames = set(f.split(".")[1] for f in final_files)
      for frame in frames:
        path_params = [base_path, folder, final_folder, geometry_folder, frame]
        if check_paths_ok(path_params):
          res.append(path_params)
    except Exception as e:
      continue
  return res

def process_single_folder(base_path, folder):
    """
    Process a single folder: figure out the final and geometry folders, 
    then check each frame. Return a list of path_params that pass the checks.
    """
    folder_path = os.path.join(base_path, folder)
    
    # If `folder` is not actually a directory or doesn't have images, skip.
    if not os.path.isdir(folder_path):
        return []
    
    try:
        images_folders = os.listdir(os.path.join(folder_path, "images"))
        # We assume there are exactly two directories: "final" and "geometry"
        # or something similar. Adjust this logic as needed.
        if "final" in images_folders[0]:
            final_folder = images_folders[0]
            geometry_folder = images_folders[1]
        else:
            final_folder = images_folders[1]
            geometry_folder = images_folders[0]
        
        final_path = os.path.join(folder_path, "images", final_folder)
        
        # Gather frames from filenames in the final folder
        final_files = os.listdir(final_path)
        frames = set(f.split(".")[1] for f in final_files if f.startswith("frame."))
        
        valid_path_params = []
        for frame in frames:
            path_params = [base_path, folder, final_folder, geometry_folder, frame]
            if check_paths_ok(path_params):
                valid_path_params.append(path_params)
        
        return valid_path_params
    
    except Exception as e:
        # You can add logging here for debugging if needed.
        return []

def list_hypersim_images(base_path):
    """
    List HyperSim images in parallel using a ThreadPoolExecutor.
    Returns a list of all valid path_params across folders.
    """
    all_folders = os.listdir(base_path)
    
    # We'll store all results in this list:
    all_results = []
    
    # Adjust max_workers as appropriate for your system / use case.
    # For CPU-bound tasks, you'd typically use ProcessPoolExecutor.
    # For I/O-bound tasks like file checks, ThreadPoolExecutor is often fine.
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Map the `process_single_folder` function to each folder
        future_to_folder = {
            executor.submit(process_single_folder, base_path, folder): folder
            for folder in all_folders
        }
        
        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                result = future.result()  # This is a list of path_params
                all_results.extend(result)
            except Exception as exc:
                print(f"Folder {folder} generated an exception: {exc}")

    return all_results

class HypersimDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.paths = list_hypersim_images(base_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        def to_tensor(path):
          return torchvision.transforms.functional.pil_to_tensor(Image.open(path).convert("RGB")) / 255.0
        return tuple(to_tensor(x) for x in make_paths(*self.paths[idx]))
       
