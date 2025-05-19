import os
import random
from PIL import Image
import torch
import torchvision
import concurrent.futures
import numpy as np

def make_paths(base_path, folder, final_folder, geometry_folder, frame):
  image = f"{base_path}/{folder}/images/{final_folder}/frame.{frame}.color.jpg"
  depth = f"{base_path}/{folder}/images/{geometry_folder}/frame.{frame}.depth_meters.png"
  return image, depth

def check_paths_ok(path_params):
  for path in make_paths(*path_params):
    try:
        Image.open(path)
    except:
        return False
  return True

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
        images_folders  = [x for x in images_folders if "cam_00" in x]
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

    
class HypersimDepthDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path, depth_path = make_paths(*self.paths[idx])
        images = torchvision.transforms.functional.pil_to_tensor(Image.open(image_path).resize((128, 128)).convert("RGB")) / 255.0
        depths = torchvision.transforms.functional.pil_to_tensor(Image.open(depth_path).resize((128, 128), 0).convert("RGB")) / 255.0
        depths = depths.norm(dim=0, keepdim=True)
        masks = depths > 0 & (depths > 1e-2) & (depths < 0.99)
        return images, depths, masks
       
import random

def get_hypersim_datasets(base_path, train_proportion=0.9):
    all_paths = sorted(list_hypersim_images(base_path), key=lambda x: "/".join(x))
    random.seed(42)
    random.shuffle(all_paths)
    training_samples = int(len(all_paths) * train_proportion)
    return HypersimDepthDataset(base_path, all_paths[:training_samples]), HypersimDepthDataset(base_path, all_paths[training_samples:])

