from elasticdino.training.ablations import train_depth, AblateDeformations, AblationTaskHeads, HypersimTaskHeads
from elasticdino.data.hypersim_depth import get_hypersim_datasets
from elasticdino.model.elasticdino import ElasticDino
import torch
import os

hypersim_path = "/mnt/home/mizrahiulysse/datasets/hypersim-depth"
   
model_config = dict(
    dino_model="b",
    n_features_in=768,
    layers={
        32: dict(hidden_features=256, n_blocks=4, layers_per_block=4),
        64: dict(hidden_features=256, n_blocks=4, layers_per_block=3),
        128: dict(hidden_features=128, n_blocks=4, layers_per_block=2),
    },
    start_size=32,
    target_size=128,
)

dino_repo = "/mnt/home/mizrahiulysse/model_cache/torch/hub/facebookresearch_dinov2_main"

def get_model(ablation):
    def get_ablated():
        if ablation == "heads":
            return ElasticDino(model_config, dino_repo)
        elif ablation == "deformations":
            return AblateDeformations(model_config, dino_repo)
        elif ablation == "none":
            return ElasticDino(model_config, dino_repo)
        else:
            raise Exception("Unknown ablation")
    
    return get_ablated


ablation = os.environ["ABLATION"]

project_folder=f"/mnt/home/mizrahiulysse/elasticdino-runs/ablations-eval/{ablation}"

checkpoints = dict(
    heads=f"/mnt/home/mizrahiulysse/elasticdino-runs/ablations/heads/2025-01-29-06:15:35/checkpoints/60000/model.safetensors",
    deformations=f"/mnt/home/mizrahiulysse/elasticdino-runs/ablations/deformations/2025-01-29-05:45:20/checkpoints/60000/model.safetensors",
    none=f"/mnt/home/mizrahiulysse/elasticdino-runs/ablations/none/2025-01-29-05:44:35/checkpoints/60000/model.safetensors",
)

train_config = dict(
  n_epochs=50,
  lr = 1e-3,
  debug_interval=100,
  n_features=64,
  save_interval=1000,
  batch_size=32,
  use_blur_loss=False,
  project_folder=project_folder,
  checkpoint=checkpoints[ablation]
)

def get_dataloaders():
    train, val = get_hypersim_datasets(hypersim_path)
    return torch.utils.data.DataLoader(train, batch_size=train_config["batch_size"], shuffle=False, num_workers=32), torch.utils.data.DataLoader(val, batch_size=train_config["batch_size"], shuffle=False, num_workers=32), 


train_depth(train_config, model_config, get_model(ablation), get_dataloaders)
