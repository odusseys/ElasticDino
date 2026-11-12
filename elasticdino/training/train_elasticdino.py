import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from elasticdino.training.losses import full_loss

#FIXME: handle imagenet missing stuff
class ElasticdinoTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, imagenet, hypersim):
        self.imagenet = imagenet
        self.hypersim = hypersim

    def __len__(self):
        return 2 * len(self.imagenet)

    def __getitem__(self, idx):
        if idx % 2 == 0:
          return self.imagenet[idx // 2]
        else:
          l = len(self.hypersim)
          idx =  (l + (idx - 1) // 2) % l
          return self.hypersim[idx]

def debug_step(batch, results, running_loss, n):
  line = f"{n} {running_loss}"
  print(line)


def train(train_config,
          model_config,
          get_models,
          get_dataloader):
  
  set_seed(42)
  kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
  dynamo_backend = "inductor"
  accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[kwargs], dynamo_backend=dynamo_backend)

  lr = train_config.get("lr", 1e-4)
  max_iterations = train_config.get("max_iterations", None)
  debug_interval = train_config.get("debug_interval", 50)
  save_interval = train_config.get("save_interval", 1000)
  start_size = model_config["start_size"]

  upscaler, task_heads = get_models()

  optimizer = torch.optim.AdamW8(
      [{"params": upscaler.parameters(), "lr": lr},
      {"params": task_heads.parameters(), "lr": lr}], eps=1e-5, weight_decay=0.0)

  dataloader = get_dataloader()
  dataloader, upscaler, task_heads, optimizer = accelerator.prepare(dataloader, upscaler, task_heads, optimizer)

  running_loss = None

  n = 0
  try:
    for epoch in range(5):
        for batch in dataloader:
            if n == max_iterations:
              return
            
            n += 1

            with accelerator.autocast():
              results = upscaler(batch["features"].to(memory_format=torch.channels_last), batch["images"].to(memory_format=torch.channels_last))
              for r in results:
                r["deformed_head_results"] = task_heads(r["deformed"])

              loss = full_loss(results, batch, start_size)
              

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if n % debug_interval == 0 and accelerator.is_local_main_process:
              debug_step(batch, results, running_loss, n)

            if running_loss is None:
              running_loss = loss.item()
            else:
              running_loss = 0.99 * running_loss + 0.01 * loss.item()

            del batch
            del loss
            del results

  except:
    del batch
    del loss
    del optimizer
    del upscaler
    del results
    raise