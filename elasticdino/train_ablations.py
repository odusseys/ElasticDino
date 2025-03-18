from elasticdino.training.ablations import train, AblateDeformations, AblationTaskHeads, HypersimTaskHeads
from elasticdino.data.hypersim import HypersimDataset
from elasticdino.model.elasticdino import ElasticDino
import torch

hypersim_path = "/mnt/home/mizrahiulysse/datasets/hypersim/"
   
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

def get_models(ablation):
    def get_ablated():
        if ablation == "heads":
            return ElasticDino(model_config), AblationTaskHeads(model_config["n_features_in"])
        elif ablation == "deformations":
            return AblateDeformations(model_config), HypersimTaskHeads(model_config["n_features_in"])
        elif ablation == "none":
            return ElasticDino(model_config), HypersimTaskHeads(model_config["n_features_in"])
        else:
            raise Exception("Unknown ablation")
    return get_ablated

import os
ablation = os.environ["ABLATION"]

project_folder=f"/mnt/home/mizrahiulysse/elasticdino-runs/ablations/{ablation}"

train_config = dict(
  n_epochs=50,
#   max_iterations=2,
  lr = 1e-4,
  debug_interval=100,
  save_interval=500,
  batch_size=16,
  use_blur_loss=False,
  project_folder=project_folder,
)

def get_dataloader():
    dataset = HypersimDataset(hypersim_path)
    return torch.utils.data.DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=16)

if __name__ == "__main__":
  train(train_config, model_config, get_models(ablation), get_dataloader)