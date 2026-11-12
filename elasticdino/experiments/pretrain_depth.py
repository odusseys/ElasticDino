

from elasticdino.model.elasticdino import ElasticDino
from elasticdino.training.depth.train_depth import make_pretraining_dataloader, train_parallel
from elasticdino.training.depth.layers import DPTDepthModel, ElasticDinoDepthModel
from elasticdino.model.dino import DinoV2
from elasticdino.data.imagenet import load_imagenet
import torch
import os
from datetime import datetime

torch.set_float32_matmul_precision('medium')

MODEL = os.environ["MODEL"]

def remove_model_prefix(d):
  """
  Recursively removes the prefix "module." from dictionary keys.

  :param d: The dictionary to process.
  :return: A new dictionary with updated keys.
  """
  if not isinstance(d, dict):
    return d

  new_dict = {}
  for key, value in d.items():
    new_key = key
    if key.startswith("module."):
      new_key = key[7:]  # Remove the prefix "model."

    # Recursively process nested dictionaries or lists
    if isinstance(value, dict):
      new_dict[new_key] = remove_model_prefix(value)
    elif isinstance(value, list):
      new_dict[new_key] = [remove_model_prefix(item) for item in value]
    else:
      new_dict[new_key] = value

  return new_dict

print(f"\n\n\n --- Starting new pretraining job ({MODEL}) --- \n\n\n")


BATCH_SIZE = 32
CHECKPOINT = None # os.environ.get("CHECKPOINT", None)


def get_model(accelerator):
    if MODEL == "DPT":
      dino = DinoV2("l")
      model = DPTDepthModel(512, dino, 128)
    elif MODEL == "ED":
      ed = ElasticDino.from_pretrained("path/to/edino", "elasticdino-32-L")
      model = ElasticDinoDepthModel(ed)
    else:
      raise Exception("Unknown model")
    if CHECKPOINT is not None:
      model.load_state_dict(remove_model_prefix(torch.load(CHECKPOINT, weights_only=True)))
    return model

def get_dataloader():
    dataloader = load_imagenet("path/to/imagenet/train", BATCH_SIZE, image_size=128)
    depth_dataloader = make_pretraining_dataloader(dataloader, "path/to/depth_anything_v2_large/", True)
    return depth_dataloader, None
    

lr = 1e-5 if CHECKPOINT is not None else 1e-4

train_config = dict(
  n_epochs=1,
  # max_iterations=10,
  lr = lr,
  decay_period=5000,
  accumulation=1,
  debug_interval=100,
  
  _interval=2000,
  display_size=128, 
  project_folder=f"elasticdino-runs/pretrain-elasticdino-depth/{MODEL}"
)

if __name__ == "__main__":
  train_parallel(get_dataloader,
            get_model,
            train_config)