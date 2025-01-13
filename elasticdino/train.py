import wandb
import os

wandb_key = os.environ["WANDB_KEY"]

def init_wandb_run(run_type, slug):
  run = wandb.init(
    project="feature-upscaling",
    config={
        "run_type": run_type,
        "slug": slug,
    }
  )
  return run


