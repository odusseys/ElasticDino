from elasticdino.training.ablations import train, AblateDeformations, AblationTaskHeads, HypersimTaskHeads, HYPERSIM_TASKS
from elasticdino.data.hypersim import HypersimDataset
from elasticdino.model.elasticdino import ElasticDino
import torch

hypersim_path = "path/to/hypersim/"
   
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

import os
ablation = os.environ["ABLATION"]
project_folder=f"elasticdino-runs/ablations/{ablation}"

head_ablation_number = None
if ablation == "heads":
    head_ablation_number = int(os.environ["HEAD_ABLATION_NUMBER"])
    project_folder += f"-loo-{head_ablation_number}"


def get_models(ablation):
    def get_ablated():
        if ablation == "heads":
            tasks = HYPERSIM_TASKS[:head_ablation_number] + HYPERSIM_TASKS[1 + head_ablation_number:]
            return ElasticDino(model_config, 'facebookresearch/dinov2'), AblationTaskHeads(model_config["n_features_in"], tasks)
        elif ablation == "deformations":
            return AblateDeformations(model_config), HypersimTaskHeads(model_config["n_features_in"])
        elif ablation == "none":
            return ElasticDino(model_config, 'facebookresearch/dinov2'), HypersimTaskHeads(model_config["n_features_in"])
        else:
            raise Exception("Unknown ablation")
    return get_ablated




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