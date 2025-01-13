from transformers import AutoModelForDepthEstimation
import torch

depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()
depth_anything = torch.compile(depth_anything)

depth_image_mean = torch.tensor([
    0.485,
    0.456,
    0.406
  ], device="cuda").reshape((1, 3, 1, 1))

depth_image_std = torch.tensor([
    0.229,
    0.224,
    0.225
  ], device="cuda").reshape((1, 3, 1, 1))

depth_size = 518

def preprocess_image_for_depth(image_tensor):
  image_tensor = (image_tensor - depth_image_mean) / depth_image_std
  image_tensor = torch.nn.functional.interpolate(
      image_tensor,
      size=(depth_size, depth_size),
      mode="bilinear",
      align_corners=False,
      antialias=True
  )
  return image_tensor


def get_depth(images):
  size = images.shape[-1]
  with torch.no_grad():
    inputs = preprocess_image_for_depth(images)
    outputs = depth_anything(pixel_values=inputs)
    predicted_depth = outputs.predicted_depth
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )
    predicted_depth[torch.isnan(predicted_depth)] = 0
    m = predicted_depth.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    M = predicted_depth.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    predicted_depth = (predicted_depth - m) / (M - m + 1e-5)
    return predicted_depth