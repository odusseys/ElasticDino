from transformers import  Mask2FormerForUniversalSegmentation
import torch
import torchvision
from PIL import Image


# load Mask2Former fine-tuned on COCO panoptic segmentation
segmentation_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic").cuda()
# segmentation_model = torch.compile(segmentation_model, mode="reduce-overhead", dynamic=True)

def make_segmentation_probabilities(outputs):
  class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
  masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

  # Scale back to preprocessed image size - (384, 384) for all models
  masks_queries_logits = torch.nn.functional.interpolate(
      masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
  )

  # Remove the null class `[..., :-1]`
  masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
  masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

  # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
  segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
  return segmentation

segmentation_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ], device="cuda").reshape((1, 3, 1, 1))

segmentation_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ], device="cuda").reshape((1, 3, 1, 1))

segmentation_image_size = 384

def preprocess_image_for_segmentation(image_tensor):
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(segmentation_image_size, segmentation_image_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  image_tensor = (image_tensor - segmentation_image_mean) / segmentation_image_std

  return image_tensor

def get_segmentation(images):
  size = images.shape[-1]
  with torch.no_grad():
    inputs = preprocess_image_for_segmentation(images)
    outputs = segmentation_model(pixel_values=inputs)
    probas = make_segmentation_probabilities(outputs)
    return torch.nn.functional.interpolate(
        probas,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )


def get_segmentation_map(image, mask, n_classes):
  image = (image * 255).to(dtype=torch.uint8)

  # make boolean segmentation masks
  mask = torch.nn.functional.interpolate(
      mask.unsqueeze(0),
      size=image.shape[-1],
      mode="nearest",
  ).squeeze(1).to(dtype=torch.long)
  one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=n_classes + 1).squeeze()
  one_hot_mask = one_hot_mask.permute(2, 0, 1).contiguous()
  boolean_mask = one_hot_mask.type(torch.bool)
  res = torchvision.utils.draw_segmentation_masks(image, boolean_mask).permute(1, 2, 0)
  return Image.fromarray(res.cpu().numpy())
