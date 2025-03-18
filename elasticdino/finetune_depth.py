

from elasticdino.model.elasticdino import ElasticDino
from elasticdino.training.depth.train_depth import make_pretraining_dataloader, train_parallel
from elasticdino.training.depth.layers import DPTDepthModel, ElasticDinoDepthModel
from elasticdino.training.depth.benchmarks import get_diode_dataloader, get_nyu_dataloader
from elasticdino.model.dino import DinoV2
from elasticdino.data.imagenet import load_imagenet
import torch
import os
from datetime import datetime

torch.set_float32_matmul_precision('medium')

MODEL = os.environ["MODEL"]
CHECKPOINT = os.environ["MODEL"]
DATASET = os.environ["DATASET"]
TRAIN_DATASET = os.environ["TRAIN_DATASET"]
VAL_DATASET = os.environ["VAL_DATASET"]


if MODEL == "DPT":
  dino = DinoV2("l")
  model = DPTDepthModel(512, dino, 128)
elif MODEL == "ED":
  ed = ElasticDino.from_pretrained("/mnt/home/mizrahiulysse/pixelvit-32-L.pth", "elasticdino-32-L")
  model = ElasticDinoDepthModel(512, ed)
elif MODEL == "ED64":
  ed = ElasticDino.from_pretrained("/mnt/home/mizrahiulysse/pixelvit-64-L.pth", "elasticdino-64-L")
  model = ElasticDinoDepthModel(512, ed)
else:
  raise Exception("Unknown model")

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


BATCH_SIZE = 16

def get_model(checkpoint, accelerator):
    if MODEL == "DPT":
      dino = DinoV2("l")
      model = DPTDepthModel(512, dino, 128)
    elif MODEL == "ED":
      ed = ElasticDino.from_pretrained("/mnt/home/mizrahiulysse/pixelvit-32-L.pth", "elasticdino-32-L")
      model = ElasticDinoDepthModel(512, ed)
    else:
      raise Exception("Unknown model")
    return model

def get_dataloaders():
    if DATASET == "nyu":
        load_func = get_nyu_dataloader
    elif DATASET == "diode":
        load_func = get_diode_dataloader
    elif DATASET == "hypersim":
        raise Exception("NYI")
    else:
        raise Exception("Unknown dataset")
    return lambda : load_func(TRAIN_DATASET, BATCH_SIZE, 128), load_func(VAL_DATASET, BATCH_SIZE, 128)
    

      
      

lr = 1e-5 if CHECKPOINT is not None else 1e-4

train_config = dict(
  n_epochs=1,
  # max_iterations=10,
  lr = lr,
  decay_period=5000,
  accumulation=1,
  debug_interval=100,
  save_interval=2000,
  display_size=128, 
  project_folder=f"/mnt/home/mizrahiulysse/elasticdino-runs/finetune-elasticdino-depth/{DATASET}/{MODEL}",
  checkpoint=CHECKPOINT
)

if __name__ == "__main__":
  train_parallel(get_dataloaders,
            get_model,
            train_config)