from elasticdino.training.ablations import train_depth, AblateDeformations, AblationTaskHeads, HypersimTaskHeads, HYPERSIM_TASKS
from elasticdino.data.hypersim_depth import get_hypersim_datasets
from elasticdino.model.elasticdino import ElasticDino
import torch
import os

hypersim_path = "path/to/hypersim"
   
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

dino_repo = "path/to/dino"

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
project_folder=f"elasticdino-runs/ablations-eval/{ablation}"

head_ablation_number = None
if ablation == "heads":
    head_ablation_number = int(os.environ["HEAD_ABLATION_NUMBER"])
    project_folder += f"-loo-{head_ablation_number}"
    ablation += f"-loo-{head_ablation_number}"

def get_model(ablation):
    def get_ablated():
        if ablation.startswith("heads"):
            tasks = HYPERSIM_TASKS[:head_ablation_number] + HYPERSIM_TASKS[1 + head_ablation_number:][:head_ablation_number]
            return ElasticDino(model_config, 'facebookresearch/dinov2'), AblationTaskHeads(model_config["n_features_in"], tasks)
        elif ablation == "deformations":
            return AblateDeformations(model_config), HypersimTaskHeads(model_config["n_features_in"])
        elif ablation == "none":
            return ElasticDino(model_config, 'facebookresearch/dinov2'), HypersimTaskHeads(model_config["n_features_in"])
        else:
            raise Exception("Unknown ablation")
    return get_ablated

checkpoints = {
    "heads-loo-0": "path/to/trained/checkpoint",
    "heads-loo-1": "path/to/trained/checkpoint",
    "heads-loo-2": "path/to/trained/checkpoint",
    "heads-loo-3": "path/to/trained/checkpoint",
    "heads-loo-4": "path/to/trained/checkpoint",
    "deformations":"path/to/trained/checkpoint",
    "none":"path/to/trained/checkpoint",
}

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
