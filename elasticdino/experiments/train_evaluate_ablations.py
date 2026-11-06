"""Train an ablated upscaler, then run the JAFAR depth probe on Hypersim."""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm.auto import tqdm

from elasticdino.model.elasticdino import ElasticDino
from elasticdino.training.ablations import (
    HYPERSIM_TASKS,
    AblateDeformations,
    AblationTaskHeads,
    HypersimTaskHeads,
    ablation_loss,
)


DEPTH_ANYTHING_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
TRAINING_LOSS_INTERVAL = 100
DEPTH_VALIDATION_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 2
CROSS_RESOLUTION_LOSS = True

MODEL_CONFIG = dict(
    dino_model="s",
    n_features_in=384,
    use_registers=False,
    layers={
        32: dict(hidden_features=256, n_blocks=4, layers_per_block=4),
        64: dict(hidden_features=256, n_blocks=4, layers_per_block=3),
        128: dict(hidden_features=128, n_blocks=4, layers_per_block=2),
    },
    start_size=32,
    target_size=128,
)


def list_hypersim_samples(root):
    """Return the cam_00 samples used by the previous Hypersim depth split."""
    root = Path(root)
    samples = []
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images_dir = scene_dir / "images"
        if not images_dir.is_dir():
            continue

        for final_dir in sorted(images_dir.glob("scene_cam_00_final_preview")):
            geometry_dir = images_dir / final_dir.name.replace("_final_preview", "_geometry_preview")
            if not geometry_dir.is_dir():
                continue

            for color_path in sorted(final_dir.glob("frame.*.color.jpg")):
                frame = color_path.name.split(".")[1]
                paths = (
                    color_path,
                    final_dir / f"frame.{frame}.diffuse_illumination.jpg",
                    final_dir / f"frame.{frame}.diffuse_reflectance.jpg",
                    final_dir / f"frame.{frame}.residual.jpg",
                    geometry_dir / f"frame.{frame}.normal_bump_cam.png",
                )
                if all(path.is_file() for path in paths):
                    samples.append(paths)
    if not samples:
        raise RuntimeError(f"No complete Hypersim cam_00 samples found under {root}")
    return samples


def load_rgb(path, image_size=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            image = T.functional.resize(
                image,
                (image_size, image_size),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
        return T.functional.pil_to_tensor(image).float() / 255.0


class HypersimAblationDataset(Dataset):
    def __init__(self, samples, image_size=MODEL_CONFIG["target_size"]):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return tuple(load_rgb(path, self.image_size) for path in self.samples[index])


class HypersimDepthProbeDataset(Dataset):
    def __init__(self, samples, image_size=MODEL_CONFIG["target_size"]):
        self.samples = samples
        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with Image.open(self.samples[index][0]) as image:
            return self.transform(image.convert("RGB"))


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def split_hypersim_samples(samples, train_proportion=0.9):
    shuffled_samples = list(samples)
    random.Random(42).shuffle(shuffled_samples)
    split_index = int(train_proportion * len(shuffled_samples))
    return shuffled_samples[:split_index], shuffled_samples[split_index:]


def make_upscaler_and_heads(ablation, head_ablation_number=None, head_ablation_count=None):
    if ablation == "deformations":
        return AblateDeformations(MODEL_CONFIG), HypersimTaskHeads(MODEL_CONFIG["n_features_in"])
    if ablation == "none":
        return ElasticDino(MODEL_CONFIG, "facebookresearch/dinov2"), HypersimTaskHeads(
            MODEL_CONFIG["n_features_in"]
        )
    if ablation == "heads":
        if head_ablation_number is not None and head_ablation_count is not None:
            raise ValueError("heads ablation accepts either a head number or a cumulative head count, not both")
        if head_ablation_count is not None:
            if not 1 <= head_ablation_count < len(HYPERSIM_TASKS):
                raise ValueError(
                    f"cumulative heads ablation requires a count in [1, {len(HYPERSIM_TASKS) - 1}]"
                )
            tasks = HYPERSIM_TASKS[head_ablation_count:]
            return ElasticDino(MODEL_CONFIG, "facebookresearch/dinov2"), AblationTaskHeads(
                MODEL_CONFIG["n_features_in"], tasks
            )
        if head_ablation_number is None or not 0 <= head_ablation_number < len(HYPERSIM_TASKS):
            raise ValueError(f"heads ablation requires a head number in [0, {len(HYPERSIM_TASKS) - 1}]")
        tasks = HYPERSIM_TASKS[:head_ablation_number] + HYPERSIM_TASKS[head_ablation_number + 1 :]
        return ElasticDino(MODEL_CONFIG, "facebookresearch/dinov2"), AblationTaskHeads(
            MODEL_CONFIG["n_features_in"], tasks
        )
    raise ValueError(f"Unknown ablation: {ablation}")


def checkpoint_state(accelerator, model, exclude_dino=False):
    state = accelerator.get_state_dict(model)
    if exclude_dino:
        state = {key: value for key, value in state.items() if not key.startswith("dino.")}
    return {key: value.detach().cpu() for key, value in state.items()}


def save_upscaler_checkpoint(
    path,
    accelerator,
    upscaler,
    task_heads,
    step,
    epoch=None,
    validation_loss=None,
    optimizer=None,
    ema_loss=None,
    best_validation_loss=None,
    epochs_without_improvement=0,
    epoch_complete=False,
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save(
            {
                "step": step,
                "epoch": epoch,
                "epoch_complete": epoch_complete,
                "validation_loss": validation_loss,
                "ema_loss": ema_loss,
                "best_validation_loss": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "model_config": MODEL_CONFIG,
                "upscaler": checkpoint_state(accelerator, upscaler, exclude_dino=True),
                "task_heads": checkpoint_state(accelerator, task_heads),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
            },
            path,
        )


def append_record(path, record):
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")


def write_run_state(run_dir, stage):
    state_path = run_dir / "run_state.json"
    temporary_path = run_dir / "run_state.tmp"
    with open(temporary_path, "w") as file:
        json.dump({"stage": stage, "updated_at": datetime.now().isoformat()}, file, indent=2)
    os.replace(temporary_path, state_path)


def get_run_stage(run_dir):
    state_path = run_dir / "run_state.json"
    if state_path.is_file():
        with open(state_path) as file:
            stage = json.load(file).get("stage")
        if stage == "complete":
            return stage

    if (run_dir / "last_depth_probe.pt").is_file():
        return "complete"
    if (run_dir / "last_upscaler.pt").is_file():
        return "depth"
    if state_path.is_file() and stage in {"upscaler", "depth"}:
        return stage
    if (run_dir / "best_depth_probe.pt").is_file() or any(
        (run_dir / "checkpoints").glob("depth-*.pt")
    ):
        return "depth"
    return "upscaler"


def latest_stage_checkpoint(run_dir, stage):
    checkpoint_dir = run_dir / "checkpoints"
    candidates = list(checkpoint_dir.glob(f"{stage}-*.pt"))
    best_name = "best_upscaler.pt" if stage == "upscaler" else "best_depth_probe.pt"
    best_path = run_dir / best_name
    if best_path.is_file():
        candidates.append(best_path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_is_compatible(run_dir, args):
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return False
    with open(config_path) as file:
        config = json.load(file)

    saved_model_config = config.get("model_config")
    if saved_model_config is None:
        checkpoint_paths = [
            run_dir / "last_upscaler.pt",
            run_dir / "best_upscaler.pt",
            latest_stage_checkpoint(run_dir, "upscaler"),
        ]
        checkpoint_path = next(
            (path for path in checkpoint_paths if path is not None and path.is_file()),
            None,
        )
        if checkpoint_path is None:
            return False
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved_model_config = checkpoint.get("model_config")

    expected = {
        "ablation": args.ablation,
        "head_ablation_number": args.head_ablation_number,
        "head_ablation_count": args.head_ablation_count,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "test": args.test,
    }
    actual = {
        "ablation": config.get("ablation"),
        "head_ablation_number": config.get("head_ablation_number"),
        "head_ablation_count": config.get("head_ablation_count"),
        "dataset_root": str(Path(config.get("dataset_root", "")).resolve()),
        "test": config.get("test", False),
    }
    normalized_model_config = json.loads(json.dumps(MODEL_CONFIG))
    return saved_model_config == normalized_model_config and actual == expected


def prepare_upscaler_for_depth(args, accelerator, run_dir):
    best_checkpoint_path = run_dir / "best_upscaler.pt"
    if not best_checkpoint_path.is_file():
        raise RuntimeError(f"Cannot resume depth stage without {best_checkpoint_path}")
    upscaler, _ = make_upscaler_and_heads(
        args.ablation,
        args.head_ablation_number,
        args.head_ablation_count,
    )
    upscaler = accelerator.prepare(upscaler)
    checkpoint = load_upscaler_checkpoint(best_checkpoint_path, accelerator, upscaler)
    upscaler.eval()
    upscaler.requires_grad_(False)
    accelerator.print(
        f"Loaded best upscaler checkpoint: {best_checkpoint_path} "
        f"(validation_loss={checkpoint.get('validation_loss')})"
    )
    return upscaler


def update_early_stopping(validation_loss, best_validation_loss, epochs_without_improvement):
    is_best = validation_loss < best_validation_loss
    if is_best:
        return True, validation_loss, 0
    return False, best_validation_loss, epochs_without_improvement + 1


def make_ablation_batch(values):
    images, diffuse_illuminations, diffuse_reflectances, residuals, normal_bump_cams = values
    return {
        "images": images,
        "diffuse_illuminations": diffuse_illuminations,
        "diffuse_reflectances": diffuse_reflectances,
        "residuals": residuals,
        "normal_bump_cams": normal_bump_cams,
    }


@torch.inference_mode()
def evaluate_upscaler(upscaler, task_heads, loader, accelerator, max_steps=None):
    upscaler.eval()
    task_heads.eval()
    total_loss = 0.0
    sample_count = 0
    progress = tqdm(
        loader,
        desc="upscaler validation",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        leave=False,
    )
    for batch_index, values in enumerate(progress):
        batch = make_ablation_batch(values)
        with torch.autocast("cuda", dtype=torch.float16):
            output = upscaler(
                batch["images"],
                return_all_scales=True,
                return_original_features=True,
            )
            results = output["all_scales"]
            loss = ablation_loss(
                [(task_heads(features), features) for features in results],
                batch,
                output["original_features"],
                MODEL_CONFIG["start_size"],
                use_blur=False,
                cross_resolution=CROSS_RESOLUTION_LOSS,
            )
        losses = accelerator.gather_for_metrics(
            loss.detach().repeat(batch["images"].shape[0])
        )
        total_loss += losses.sum().item()
        sample_count += losses.numel()
        if max_steps and batch_index + 1 >= max_steps:
            break
    return total_loss / sample_count


def load_upscaler_checkpoint(path, accelerator, upscaler):
    accelerator.wait_for_everyone()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = accelerator.unwrap_model(upscaler)
    missing, unexpected = model.load_state_dict(checkpoint["upscaler"], strict=False)
    if unexpected or any(not key.startswith("dino.") for key in missing):
        raise RuntimeError(f"Could not load best upscaler checkpoint: missing={missing}, unexpected={unexpected}")
    return checkpoint


def resume_upscaler_checkpoint(path, accelerator, upscaler, task_heads, optimizer):
    checkpoint = load_upscaler_checkpoint(path, accelerator, upscaler)
    accelerator.unwrap_model(task_heads).load_state_dict(checkpoint["task_heads"])
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        accelerator.print(f"Checkpoint {path} has no optimizer state; resuming with a fresh optimizer")
    return checkpoint


def train_upscaler(
    args,
    accelerator,
    train_samples,
    val_samples,
    run_dir,
    training_loss_path,
    validation_loss_path,
    resume_checkpoint=None,
):
    upscaler, task_heads = make_upscaler_and_heads(
        args.ablation,
        args.head_ablation_number,
        args.head_ablation_count,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": upscaler.parameters(), "lr": args.upscaler_lr},
            {"params": task_heads.parameters(), "lr": args.upscaler_lr},
        ],
        eps=1e-5,
        weight_decay=0.0,
    )
    train_loader = make_loader(
        HypersimAblationDataset(train_samples),
        args.upscaler_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        HypersimAblationDataset(val_samples),
        args.upscaler_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, upscaler, task_heads, optimizer = accelerator.prepare(
        train_loader,
        val_loader,
        upscaler,
        task_heads,
        optimizer,
    )

    step = 0
    ema_loss = None
    stop = False
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = run_dir / "best_upscaler.pt"
    start_epoch = 0
    validation_loss = float("inf")
    if resume_checkpoint is not None:
        checkpoint = resume_upscaler_checkpoint(
            resume_checkpoint,
            accelerator,
            upscaler,
            task_heads,
            optimizer,
        )
        step = checkpoint.get("step", 0)
        ema_loss = checkpoint.get("ema_loss")
        validation_loss = checkpoint.get("validation_loss") or float("inf")
        best_validation_loss = checkpoint.get("best_validation_loss") or validation_loss
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        checkpoint_epoch = checkpoint.get("epoch") or 1
        start_epoch = checkpoint_epoch if checkpoint.get("epoch_complete", False) else checkpoint_epoch - 1
        accelerator.print(
            f"Resuming upscaler from {resume_checkpoint} at epoch {start_epoch + 1}, step {step}"
        )

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.upscaler_epochs):
        upscaler.train()
        task_heads.train()
        progress = tqdm(
            train_loader,
            desc=f"upscaler epoch {epoch + 1}/{args.upscaler_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
        for batch_index, values in enumerate(progress):
            if args.max_upscaler_steps and step >= args.max_upscaler_steps:
                stop = True
                break
            step += 1
            batch = make_ablation_batch(values)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                output = upscaler(
                    batch["images"],
                    return_all_scales=True,
                    return_original_features=True,
                )
                results = output["all_scales"]
                loss = ablation_loss(
                    [(task_heads(features), features) for features in results],
                    batch,
                    output["original_features"],
                    MODEL_CONFIG["start_size"],
                    use_blur=False,
                    cross_resolution=CROSS_RESOLUTION_LOSS,
                )
            accelerator.backward(loss)
            optimizer.step()

            loss_value = loss.detach().item()
            ema_loss = loss_value if ema_loss is None else 0.97 * ema_loss + 0.03 * loss_value
            progress.set_postfix(step=step, ema_loss=f"{ema_loss:.6f}")
            if step % args.training_loss_interval == 0:
                if accelerator.is_main_process:
                    append_record(training_loss_path, {"step": step, "ema_loss": ema_loss})
            if args.save_interval and step % args.save_interval == 0:
                save_upscaler_checkpoint(
                    run_dir / "checkpoints" / f"upscaler-{step}.pt",
                    accelerator,
                    upscaler,
                    task_heads,
                    step,
                    epoch=epoch + 1,
                    optimizer=optimizer,
                    ema_loss=ema_loss,
                    best_validation_loss=best_validation_loss,
                    epochs_without_improvement=epochs_without_improvement,
                )
            if args.max_upscaler_steps and step >= args.max_upscaler_steps:
                stop = True
                break
        validation_loss = evaluate_upscaler(
            upscaler,
            task_heads,
            val_loader,
            accelerator,
            max_steps=args.max_validation_steps,
        )
        is_best, best_validation_loss, epochs_without_improvement = update_early_stopping(
            validation_loss,
            best_validation_loss,
            epochs_without_improvement,
        )
        if is_best:
            save_upscaler_checkpoint(
                best_checkpoint_path,
                accelerator,
                upscaler,
                task_heads,
                step,
                epoch=epoch + 1,
                validation_loss=validation_loss,
                optimizer=optimizer,
                ema_loss=ema_loss,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                epoch_complete=True,
            )
        if accelerator.is_main_process:
            append_record(
                validation_loss_path,
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "val_loss": validation_loss,
                    "is_best": is_best,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )
            progress.write(
                f"upscaler epoch={epoch + 1} val_loss={validation_loss:.6f} "
                f"best={is_best} patience={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )
        save_upscaler_checkpoint(
            run_dir / "checkpoints" / f"upscaler-{step}.pt",
            accelerator,
            upscaler,
            task_heads,
            step,
            epoch=epoch + 1,
            validation_loss=validation_loss,
            optimizer=optimizer,
            ema_loss=ema_loss,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
            epoch_complete=True,
        )
        early_stop = epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        if early_stop:
            accelerator.print(
                f"Stopping upscaler training after {EARLY_STOPPING_PATIENCE} "
                "epochs without validation improvement"
            )
        if stop or early_stop:
            break

    save_upscaler_checkpoint(
        run_dir / "last_upscaler.pt",
        accelerator,
        upscaler,
        task_heads,
        step,
        epoch=epoch + 1,
        validation_loss=validation_loss,
        optimizer=optimizer,
        ema_loss=ema_loss,
        best_validation_loss=best_validation_loss,
        epochs_without_improvement=epochs_without_improvement,
        epoch_complete=True,
    )
    load_upscaler_checkpoint(best_checkpoint_path, accelerator, upscaler)
    accelerator.print(
        f"Using best upscaler checkpoint: {best_checkpoint_path} "
        f"(validation_loss={best_validation_loss:.6f})"
    )
    del task_heads, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    upscaler.eval()
    upscaler.requires_grad_(False)
    return upscaler


class DepthProbe(nn.Module):
    def __init__(self, feature_dim, num_bins=256):
        super().__init__()
        self.classifier = nn.Conv2d(2 * feature_dim, num_bins, kernel_size=1)

    def forward(self, features, class_token, output_size):
        class_token = F.normalize(class_token.float(), dim=1)[:, :, None, None]
        features = torch.cat([features.float(), class_token.expand_as(features)], dim=1)
        logits = self.classifier(features)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear")
        return logits


# Adapted from JAFAR evaluation/train_probes.py and src/loss.py at commit
# 25321a2def9b00d06a8a8794ca0a799dcfd18d86 (Apache-2.0).
class JafarSigLoss(nn.Module):
    def __init__(self, max_depth=10.0, warm_iter=100):
        super().__init__()
        self.max_depth = max_depth
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.eps = 0.001

    def forward(self, prediction, target):
        mask = (target > 0) & (target <= self.max_depth)
        prediction = prediction[mask]
        target = target[mask]
        difference = torch.log(prediction + self.eps) - torch.log(target + self.eps)
        if self.warm_up_counter < self.warm_iter:
            self.warm_up_counter += 1
            return torch.sqrt(0.15 * difference.mean().square())
        return torch.sqrt(torch.var(difference) + 0.15 * difference.mean().square())


class JafarGradientLoss(nn.Module):
    def __init__(self, max_depth=10.0, weight=0.5):
        super().__init__()
        self.max_depth = max_depth
        self.weight = weight
        self.eps = 0.001

    def forward(self, prediction, target):
        predictions = [prediction] + [prediction[:, :, :: 2 * i, :: 2 * i] for i in range(1, 4)]
        targets = [target] + [target[:, :, :: 2 * i, :: 2 * i] for i in range(1, 4)]
        loss = prediction.new_tensor(0.0)
        for current_prediction, current_target in zip(predictions, targets):
            mask = (current_target > 0) & (current_target <= self.max_depth)
            count = mask.sum()
            log_difference = (
                torch.log(current_prediction + self.eps) - torch.log(current_target + self.eps)
            ) * mask
            vertical = torch.abs(log_difference[0:-2, :] - log_difference[2:, :])
            vertical = vertical * (mask[0:-2, :] * mask[2:, :])
            horizontal = torch.abs(log_difference[:, 0:-2] - log_difference[:, 2:])
            horizontal = horizontal * (mask[:, 0:-2] * mask[:, 2:])
            loss = loss + (horizontal.sum() + vertical.sum()) / count
        return self.weight * loss


class DepthAnythingTeacher:
    def __init__(self, model_name, device):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def predict(self, images, output_size):
        uint8_images = (255 * images.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
        inputs = self.processor(images=uint8_images, return_tensors="pt").to(self.device)
        pseudo_depth = self.model(**inputs).predicted_depth
        return F.interpolate(
            pseudo_depth.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


def predict_depth(upscaler, probe, images, min_depth, max_depth):
    with torch.no_grad():
        output = upscaler(images, return_class_token=True)
        features = output["deformed_features"]
        class_token = output["class_token"]
    logits = probe(features, class_token, images.shape[-2:])
    bins = torch.linspace(min_depth, max_depth, logits.shape[1], device=logits.device)
    probabilities = F.relu(logits) + 0.1
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    return torch.einsum("bkhw,k->bhw", probabilities, bins).unsqueeze(1)


def eval_depth_metrics(target, prediction, min_depth=1e-3, max_depth=10.0):
    mask = (target > min_depth) & (target < max_depth)
    target = target[mask]
    prediction = prediction[mask]
    threshold = np.maximum(target / prediction, prediction / target)
    log_error = np.log(prediction) - np.log(target)
    return {
        "d1": (threshold < 1.25).mean(),
        "d2": (threshold < 1.25**2).mean(),
        "d3": (threshold < 1.25**3).mean(),
        "abs_rel": np.mean(np.abs(target - prediction) / target),
        "sq_rel": np.mean((target - prediction) ** 2 / target),
        "rmse": np.sqrt(np.mean((target - prediction) ** 2)),
        "rmse_log": np.sqrt(np.mean((np.log(target) - np.log(prediction)) ** 2)),
        "log_10": np.mean(np.abs(np.log10(target) - np.log10(prediction))),
        "silog": np.sqrt(np.mean(log_error**2) - np.mean(log_error) ** 2) * 100,
    }


@torch.inference_mode()
def evaluate_depth(upscaler, probe, teacher, loader, accelerator, args):
    probe.eval()
    totals = {name: 0.0 for name in ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log_10", "silog")}
    sample_count = 0
    total_loss = 0.0
    loss_count = 0
    sigloss = JafarSigLoss(args.max_depth, warm_iter=0)
    gradient_loss = JafarGradientLoss(args.max_depth)
    progress = tqdm(
        loader,
        desc="depth validation",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        leave=False,
    )
    for batch_index, images in enumerate(progress):
        target = teacher.predict(images, images.shape[-2:])
        prediction = predict_depth(upscaler, probe, images, args.min_depth, args.max_depth)
        loss = sigloss(prediction, target) + gradient_loss(prediction, target)
        prediction, target, losses = accelerator.gather_for_metrics(
            (prediction, target, loss.reshape(1))
        )
        total_loss += losses.sum().item()
        loss_count += losses.numel()
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        for index in range(target.shape[0]):
            current = eval_depth_metrics(
                target[index : index + 1],
                prediction[index : index + 1],
                args.min_depth,
                args.max_depth,
            )
            for name, value in current.items():
                totals[name] += float(value)
            sample_count += 1
        if args.max_validation_steps and batch_index + 1 >= args.max_validation_steps:
            break
    metrics = {name: value / sample_count for name, value in totals.items()}
    return metrics, total_loss / loss_count


def save_depth_checkpoint(
    path,
    accelerator,
    probe,
    optimizer,
    step,
    epoch,
    validation_loss,
    metrics,
    scheduler=None,
    ema_loss=None,
    best_validation_loss=None,
    best_metrics=None,
    epochs_without_improvement=0,
    history=None,
    epoch_complete=False,
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save(
            {
                "step": step,
                "epoch": epoch,
                "epoch_complete": epoch_complete,
                "validation_loss": validation_loss,
                "depth_probe": checkpoint_state(accelerator, probe),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "ema_loss": ema_loss,
                "best_validation_loss": best_validation_loss,
                "best_metrics": best_metrics,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "metrics": metrics,
            },
            path,
        )


def load_depth_checkpoint(path, accelerator, probe):
    accelerator.wait_for_everyone()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    accelerator.unwrap_model(probe).load_state_dict(checkpoint["depth_probe"])
    return checkpoint


def resume_depth_checkpoint(path, accelerator, probe, optimizer, scheduler):
    checkpoint = load_depth_checkpoint(path, accelerator, probe)
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def train_depth_probe(
    args,
    accelerator,
    upscaler,
    train_samples,
    val_samples,
    run_dir,
    training_loss_path,
    validation_path,
    resume_checkpoint=None,
):
    train_dataset = HypersimDepthProbeDataset(train_samples, args.depth_image_size)
    val_dataset = HypersimDepthProbeDataset(val_samples, args.depth_image_size)
    train_loader = make_loader(
        train_dataset,
        args.depth_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        DEPTH_VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )

    probe = DepthProbe(MODEL_CONFIG["n_features_in"])
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.depth_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.depth_epochs * len(train_loader),
    )
    probe, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        probe, optimizer, scheduler, train_loader, val_loader
    )
    teacher = DepthAnythingTeacher(args.depth_anything_model, accelerator.device)
    sigloss = JafarSigLoss(args.max_depth)
    gradient_loss = JafarGradientLoss(args.max_depth)

    step = 0
    history = []
    stop = False
    ema_loss = None
    best_validation_loss = float("inf")
    best_metrics = None
    epochs_without_improvement = 0
    best_checkpoint_path = run_dir / "best_depth_probe.pt"
    start_epoch = 0
    validation_loss = float("inf")
    metrics = None
    if resume_checkpoint is not None:
        checkpoint = resume_depth_checkpoint(
            resume_checkpoint,
            accelerator,
            probe,
            optimizer,
            scheduler,
        )
        step = checkpoint.get("step", 0)
        ema_loss = checkpoint.get("ema_loss")
        validation_loss = checkpoint.get("validation_loss") or float("inf")
        metrics = checkpoint.get("metrics")
        best_validation_loss = checkpoint.get("best_validation_loss") or validation_loss
        best_metrics = checkpoint.get("best_metrics") or metrics
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        history = checkpoint.get("history") or []
        checkpoint_epoch = checkpoint.get("epoch") or 1
        start_epoch = checkpoint_epoch if checkpoint.get("epoch_complete", False) else checkpoint_epoch - 1
        accelerator.print(
            f"Resuming depth probe from {resume_checkpoint} at epoch {start_epoch + 1}, step {step}"
        )

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.depth_epochs):
        probe.train()
        total_loss = 0.0
        processed_batches = 0
        progress = tqdm(
            train_loader,
            desc=f"depth epoch {epoch + 1}/{args.depth_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
        for batch_index, images in enumerate(progress):
            if args.max_depth_steps and step >= args.max_depth_steps:
                stop = True
                break
            if random.random() < 0.5:
                images = torch.flip(images, dims=[3])
            optimizer.zero_grad()
            target = teacher.predict(images, images.shape[-2:])
            prediction = predict_depth(upscaler, probe, images, args.min_depth, args.max_depth)
            loss = sigloss(prediction, target) + gradient_loss(prediction, target)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            step += 1
            processed_batches += 1
            loss_value = loss.detach().item()
            total_loss += loss_value
            ema_loss = loss_value if ema_loss is None else 0.97 * ema_loss + 0.03 * loss_value
            progress.set_postfix(
                step=step,
                ema_loss=f"{ema_loss:.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if step % args.training_loss_interval == 0:
                if accelerator.is_main_process:
                    append_record(training_loss_path, {"step": step, "ema_loss": ema_loss})
            if args.save_interval and step % args.save_interval == 0:
                save_depth_checkpoint(
                    run_dir / "checkpoints" / f"depth-{step}.pt",
                    accelerator,
                    probe,
                    optimizer,
                    step,
                    epoch + 1,
                    validation_loss,
                    metrics,
                    scheduler=scheduler,
                    ema_loss=ema_loss,
                    best_validation_loss=best_validation_loss,
                    best_metrics=best_metrics,
                    epochs_without_improvement=epochs_without_improvement,
                    history=history,
                )
            if args.max_depth_steps and step >= args.max_depth_steps:
                stop = True
                break

        metrics, validation_loss = evaluate_depth(
            upscaler, probe, teacher, val_loader, accelerator, args
        )
        is_best, best_validation_loss, epochs_without_improvement = update_early_stopping(
            validation_loss,
            best_validation_loss,
            epochs_without_improvement,
        )
        if is_best:
            best_metrics = metrics
            save_depth_checkpoint(
                best_checkpoint_path,
                accelerator,
                probe,
                optimizer,
                step,
                epoch + 1,
                validation_loss,
                metrics,
                scheduler=scheduler,
                ema_loss=ema_loss,
                best_validation_loss=best_validation_loss,
                best_metrics=best_metrics,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                epoch_complete=True,
            )
        if accelerator.is_main_process:
            append_record(
                validation_path,
                {"epoch": epoch + 1, "step": step, "val_loss": validation_loss, **metrics},
            )
            progress.write(
                f"depth epoch={epoch + 1} val_loss={validation_loss:.6f} "
                f"metrics={json.dumps(metrics)}"
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": total_loss / processed_batches if processed_batches else ema_loss,
                "validation_loss": validation_loss,
                "metrics": metrics,
                "is_best": is_best,
                "epochs_without_improvement": epochs_without_improvement,
            }
            history.append(record)
            print(json.dumps(record, indent=2))
            with open(run_dir / "depth_metrics.json", "w") as file:
                json.dump(history, file, indent=2)
        save_depth_checkpoint(
            run_dir / "checkpoints" / f"depth-{step}.pt",
            accelerator,
            probe,
            optimizer,
            step,
            epoch + 1,
            validation_loss,
            metrics,
            scheduler=scheduler,
            ema_loss=ema_loss,
            best_validation_loss=best_validation_loss,
            best_metrics=best_metrics,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            epoch_complete=True,
        )
        early_stop = epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        if early_stop:
            accelerator.print(
                f"Stopping depth training after {EARLY_STOPPING_PATIENCE} "
                "epochs without validation improvement"
            )
        if stop or early_stop:
            break

    save_depth_checkpoint(
        run_dir / "last_depth_probe.pt",
        accelerator,
        probe,
        optimizer,
        step,
        epoch + 1,
        validation_loss,
        metrics,
        scheduler=scheduler,
        ema_loss=ema_loss,
        best_validation_loss=best_validation_loss,
        best_metrics=best_metrics,
        epochs_without_improvement=epochs_without_improvement,
        history=history,
        epoch_complete=True,
    )
    load_depth_checkpoint(best_checkpoint_path, accelerator, probe)
    accelerator.print(
        f"Using best depth checkpoint: {best_checkpoint_path} "
        f"(validation_loss={best_validation_loss:.6f})"
    )
    return best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Path to the Hypersim dataset")
    parser.add_argument("--output-root", default="elasticdino-runs/ablations-combined")
    parser.add_argument("--ablation", choices=("none", "heads", "deformations"), default=os.getenv("ABLATION", "none"))
    head_ablation = parser.add_mutually_exclusive_group()
    head_ablation.add_argument(
        "--head-ablation-number",
        type=int,
        default=int(os.environ["HEAD_ABLATION_NUMBER"]) if "HEAD_ABLATION_NUMBER" in os.environ else None,
    )
    head_ablation.add_argument(
        "--head-ablation-count",
        type=int,
        default=int(os.environ["HEAD_ABLATION_COUNT"]) if "HEAD_ABLATION_COUNT" in os.environ else None,
        help="cumulatively remove this many heads from the start of HYPERSIM_TASKS",
    )
    parser.add_argument("--upscaler-epochs", type=int, default=20)
    parser.add_argument("--upscaler-batch-size", type=int, default=32)
    parser.add_argument("--upscaler-lr", type=float, default=5e-4)
    parser.add_argument("--depth-epochs", type=int, default=20)
    parser.add_argument("--depth-batch-size", type=int, default=32)
    parser.add_argument("--depth-lr", type=float, default=5e-4)
    parser.add_argument("--depth-image-size", type=int, default=MODEL_CONFIG["target_size"])
    parser.add_argument("--depth-anything-model", default=DEPTH_ANYTHING_MODEL)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--training-loss-interval", type=int, default=TRAINING_LOSS_INTERVAL)
    parser.add_argument("--max-upscaler-steps", type=int)
    parser.add_argument("--max-depth-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test:
        args.upscaler_epochs = 1
        args.depth_epochs = 1
        args.max_upscaler_steps = 1
        args.max_depth_steps = 1
        args.max_validation_steps = 1
        args.save_interval = 1
        args.training_loss_interval = 1
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA")

    set_seed(42)
    accelerator = Accelerator(mixed_precision="no")
    suffix = args.ablation
    if args.ablation == "heads":
        if args.head_ablation_count is not None:
            suffix += f"-cumulative-{args.head_ablation_count}"
        else:
            suffix += f"-loo-{args.head_ablation_number}"
    if args.test:
        suffix += "-test"
    ablation_dir = Path(args.output_root) / suffix
    existing_runs = sorted(path for path in ablation_dir.glob("*") if path.is_dir())
    run_dir = None
    if args.resume and existing_runs:
        candidate = existing_runs[-1]
        if run_is_compatible(candidate, args):
            run_dir = candidate
        else:
            accelerator.print(
                f"Latest run {candidate} is incompatible with the current dataset/model; "
                "starting a new run"
            )
    is_resuming = run_dir is not None
    if run_dir is None:
        run_dir = ablation_dir / datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    stage = get_run_stage(run_dir) if is_resuming else "upscaler"
    log_dir = run_dir / "logs"
    upscaler_training_loss_path = log_dir / "upscaler_training_loss.txt"
    upscaler_validation_loss_path = log_dir / "upscaler_validation.txt"
    depth_training_loss_path = log_dir / "depth_training_loss.txt"
    depth_validation_path = log_dir / "depth_validation.txt"
    if accelerator.is_main_process:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        if not is_resuming:
            for path in (
                upscaler_training_loss_path,
                upscaler_validation_loss_path,
                depth_training_loss_path,
                depth_validation_path,
            ):
                path.write_text("")
            with open(run_dir / "config.json", "w") as file:
                json.dump({**vars(args), "model_config": MODEL_CONFIG}, file, indent=2)
            write_run_state(run_dir, stage)
    accelerator.wait_for_everyone()

    if is_resuming:
        accelerator.print(f"Resuming run {run_dir} at stage: {stage}")
    if stage == "complete":
        accelerator.print(f"Ablation already complete; skipping: {run_dir}")
        return

    samples = list_hypersim_samples(args.dataset_root)
    train_samples, val_samples = split_hypersim_samples(samples)
    accelerator.print(
        f"Loaded {len(samples)} Hypersim samples from {args.dataset_root}: "
        f"{len(train_samples)} train, {len(val_samples)} validation"
    )
    if stage == "upscaler":
        upscaler = train_upscaler(
            args,
            accelerator,
            train_samples,
            val_samples,
            run_dir,
            upscaler_training_loss_path,
            upscaler_validation_loss_path,
            resume_checkpoint=latest_stage_checkpoint(run_dir, "upscaler") if is_resuming else None,
        )
        if accelerator.is_main_process:
            write_run_state(run_dir, "depth")
        accelerator.wait_for_everyone()
    else:
        upscaler = prepare_upscaler_for_depth(args, accelerator, run_dir)

    metrics = train_depth_probe(
        args,
        accelerator,
        upscaler,
        train_samples,
        val_samples,
        run_dir,
        depth_training_loss_path,
        depth_validation_path,
        resume_checkpoint=(
            latest_stage_checkpoint(run_dir, "depth")
            if is_resuming and stage == "depth"
            else None
        ),
    )
    if accelerator.is_main_process:
        write_run_state(run_dir, "complete")
        print(f"Finished. Outputs: {run_dir}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
