# Copyright (c) 2019 Henrique Morimitsu
# This file is released under the "MIT License". Please see the LICENSE file
# included in this package.
#
# Many of the functions are modifications from the FlowNetPytorch package
# available at https://github.com/ClementPinard/FlowNetPytorch, but adapted
# to be implemented with PyTorch functions.

import numpy as np
import torch
import torch.nn.functional as F
import warnings
from abc import abstractmethod
from typing import Any, Sequence, Tuple
from PIL import Image


""" Common tranformation functions for augmenting data for training deep
optical flow models. The functions most functions take two 4D torch.Tensor
inputs: (1) a batch of images, (2) a batch of flows. The tensors are assumed
to have a shape [N, C, H, W], where:
    N = batch size,
    C = number of channels (typically 3 for images and 2 for flows),
    H = height,
    W = width.
"""

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return Image.fromarray(flow_image)


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

class BaseTransform(object):
    @abstractmethod
    def __call__(self,
                 images: Any,
                 flows: Any) -> Tuple[Any, Any]:
        pass


class ToTensor(BaseTransform):
    """ Converts a 4D numpy.ndarray or a list of 3D numpy.ndarrays into a
    4D torch.Tensor. Unlike the torchvision implementation, no normalization is
    applied to the values of the inputs.

    Args:
        images_order: str: optional, default 'HWC'
            Must be one of {'CHW', 'HWC'}. Indicates whether the input images
            have the channels first or last.
        flows_order: str: optional, default 'HWC'
            Must be one of {'CHW', 'HWC'}. Indicates whether the input flows
            have the channels first or last.
        fp16: bool: optional, default False
            If True, the tensors use have-precision floating point.
        device: str or torch.device: optional, default 'cpu'
            Name of the torch device where the tensors will be put in.
    """
    def __init__(self,
                 images_order: str = 'HWC',
                 flows_order: str = 'HWC',
                 fp16: bool = False,
                 device: Any = 'cpu') -> None:
        self.images_order = images_order.upper()
        self.flows_order = flows_order.upper()
        self.fp16 = fp16
        self.device = device

    def __call__(self,
                 images: Any,
                 flows: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, list) or isinstance(images, tuple):
            images = np.stack(images, axis=0)
        if isinstance(flows, list) or isinstance(flows, tuple):
            flows = np.stack(flows, axis=0)
        images = torch.from_numpy(images)
        flows = torch.from_numpy(flows)
        if self.images_order == 'HWC':
            images = images.permute(0, 3, 1, 2)
        if self.flows_order == 'HWC':
            flows = flows.permute(0, 3, 1, 2)
        if self.fp16:
            images = images.half()
            flows = flows.half()
        else:
            images = images.float()
            flows = flows.float()
        images = images.to(device=self.device)
        flows = flows.to(device=self.device)
        return images, flows


class Compose(BaseTransform):
    """ Similar to torchvision Compose. Applies a series of transforms from
    the input list in sequence.

    Args:
        transforms_list: Sequence[BaseTransform]:
            A sequence of transforms to be applied.
    """
    def __init__(self,
                 transforms_list: Sequence[BaseTransform]) -> None:
        self.transforms_list = transforms_list

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms_list:
            images, flows = t(images, flows)
        return images, flows


class Normalize(BaseTransform):
    """ Normalizes the images according to the equation:
    images = (images - means) / stdevs.

    Args:
        means: Sequence[float]:
            A list with the values of the means of each of the channels of the
            image.
        stdevs: Sequence[float]:
            A list with the values of the standard deviations of each of the
            channels of the image.
    """
    def __init__(self,
                 means: Sequence[float],
                 stdevs: Sequence[float]) -> None:
        self.means = torch.from_numpy(np.array(means)).reshape(1, -1, 1, 1)
        self.stdevs = torch.from_numpy(np.array(stdevs)).reshape(1, -1, 1, 1)
        self.has_converted = False

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.has_converted:
            self.means = self.means.type(images.dtype).to(images.device)
            self.stdevs = self.stdevs.type(images.dtype).to(images.device)
            self.has_converted = True
        images -= self.means
        images /= self.stdevs
        return images, flows


class Crop(BaseTransform):
    """ Crops a region at the given position of the image according to the
    given size.

    Args:
        coordinates: Tuple[int, int]:
            The (y, x) position of the origin of the crop. The coordinates
            values depend on the choice of the parameters is_center. If
            is_center == True, then coordinates should be the center point of
            the region to be cropped. Otherwise, it should be the top-left
            corner of the region.
        size: Tuple[int, int]:
            The size (height, width) of the region to be cropped.
        is_center: bool: optional, default True
            Determines how the coordinates will be used. See the comments about
            coordinates.
    """
    def __init__(self,
                 coordinates: Tuple[int, int],
                 size: Tuple[int, int],
                 is_center: bool = True) -> None:
        y, x = coordinates
        assert y >= 0 and x >= 0, 'The coordinates must be positive.'
        h, w = size
        assert h > 0 and w > 0, 'The size must be larger than zero.'
        self.coordinates = coordinates
        self.size = size
        self.is_center = is_center

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = images.shape
        ch, cw = self.size
        cy, cx = self.coordinates
        if self.is_center:
            ch2 = ch // 2
            cw2 = cw // 2
            y1 = cy - ch2
            x1 = cx - cw2
            y2 = cy + ch - ch2
            x2 = cx + cw - cw2
        else:
            y1 = cy
            x1 = cx
            y2 = y1 + ch
            x2 = x1 + cw
        if y1 < 0:
            warnings.warn('y1 < 0 when cropping. It will be set as y1 = 0.',
                          RuntimeWarning)
            y1 = 0
        if x1 < 0:
            warnings.warn('x1 < 0 when cropping. It will be set as x1 = 0.',
                          RuntimeWarning)
            x1 = 0
        if y2 > h:
            warnings.warn(
                'y2 > image_height when cropping. '
                'It will be set as y2 = image_height.', RuntimeWarning)
            y2 = h
        if x2 > w:
            warnings.warn(
                'x2 > image_width when cropping. '
                'It will be set as x2 = image_width.', RuntimeWarning)
            x2 = w
        images = images[:, :, y1:y2, x1:x2]
        flows = flows[:, :, y1:y2, x1:x2]
        return images, flows



class CenterCrop(BaseTransform):
    """ Crops a region in the center of the image according to the given size.

    Args:
        size: Tuple[int, int]:
            The size (height, width) of the region to be cropped.
    """
    def __init__(self,
                 size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = images.shape
        th, tw = self.size
        if w == tw and h == th:
            return images, flows

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        images = images[:, :, y1:y1+th, x1:x1+tw]
        flows = flows[:, :, y1:y1+th, x1:x1+tw]
        return images, flows


class Resize(BaseTransform):
    """ Resizes the inputs to a new given size.

    Args:
        size: Tuple[int, int]:
            The new size (height, width) of the inputs.
    """
    def __init__(self,
                 size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        yscale = float(self.size[0]) / images.shape[2]
        xscale = float(self.size[1]) / images.shape[3]
        images = F.interpolate(
            images, size=self.size, mode='bilinear', align_corners=False)
        flows = F.interpolate(
            flows, size=self.size, mode='bilinear', align_corners=False)
        flows[:, 0] *= yscale
        flows[:, 1] *= xscale
        return images, flows


class RandomHorizontalFlip(BaseTransform):
    """ Randomly flips the entire batch horizontally with a probability of 0.5.
    """
    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() < 0.5:
            images = torch.flip(images, dims=[3])
            flows = torch.flip(flows, dims=[3])
            flows[:, 0] *= -1
        return images, flows


class RandomVerticalFlip(BaseTransform):
    """ Randomly flips the entire batch vertically with a probability of 0.5.
    """
    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() < 0.5:
            images = torch.flip(images, dims=[2])
            flows = torch.flip(flows, dims=[2])
            flows[:, 1] *= -1
        return images, flows


class RandomAdditiveColor(BaseTransform):
    """ Adds a random value to the image. The value is sampled from a normal
    distribution.

    Args:
        stdev: float:
            The standard deviation value for the normal distribution.
        independent: bool: optional, default False
            if True, one different value is sampled for each image of the
            batch. If False, the same value is added to all images.
    """
    def __init__(self,
                 stdev: float,
                 independent: bool = False) -> None:
        self.stdev = stdev
        self.independent = independent

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.independent:
            add_vals = torch.normal(0.0, self.stdev, [images.shape[0]])
        else:
            add_vals = torch.normal(0.0, self.stdev, [1])
            add_vals = add_vals.repeat(images.shape[0])
        add_vals = add_vals.type(images.dtype)
        add_vals = add_vals.to(images.device)
        add_vals = add_vals.reshape(-1, 1, 1, 1)
        images += add_vals
        return images, flows


class RandomMultiplicativeColor(BaseTransform):
    """ Multiplies the image by a random value. The value is sampled from an
    uniform distribution according to the given boundaries.

    Args:
        range_min: float:
            The minimum value of the uniform distribution.
        range_max: float:
            The maximum value of the uniform distribution.
        independent: bool: optional, default False
            if True, one different value is sampled for each image of the
            batch. If False, all images are multiplied by the same value.
    """
    def __init__(self,
                 range_min: float,
                 range_max: float,
                 independent: bool = False) -> None:
        self.range_min = range_min
        self.range_max = range_max
        self.independent = independent
        self.interval = range_max - range_min

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.independent:
            mult_vals = torch.rand([images.shape[0]])
        else:
            mult_vals = torch.rand([1])
            mult_vals = mult_vals.repeat(images.shape[0])
        mult_vals = mult_vals.type(images.dtype)
        mult_vals = mult_vals.to(images.device)
        mult_vals *= self.interval
        mult_vals += self.range_min
        mult_vals = mult_vals.reshape(-1, 1, 1, 1)
        images *= mult_vals
        return images, flows


class RandomGammaColor(BaseTransform):
    """ Applies a power transform to the image by a random value. The value is
    sampled from an uniform distribution according to the given boundaries.

    Args:
        range_min: float:
            The minimum value of the uniform distribution.
        range_max: float:
            The maximum value of the uniform distribution.
        pixel_min: float:
            The minimum valid value of a pixel of the image.
        pixel_max: float:
            The maximum valid value of a pixel of the image.
        independent: bool: optional, default False
            if True, one different value is sampled for each image of the
            batch. If False, all images are multiplied by the same value.
    """
    def __init__(self,
                 range_min: float,
                 range_max: float,
                 pixel_min: float,
                 pixel_max: float,
                 independent: bool = False) -> None:
        self.range_min = range_min
        self.range_max = range_max
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.independent = independent
        self.interval = range_max - range_min
        self.pixel_interval = pixel_max - pixel_min

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.independent:
            expo_vals = torch.rand([images.shape[0]])
        else:
            expo_vals = torch.rand([1])
            expo_vals = expo_vals.repeat(images.shape[0])
        expo_vals = expo_vals.type(images.dtype)
        expo_vals = expo_vals.to(images.device)
        expo_vals *= self.interval
        expo_vals += self.range_min
        expo_vals = torch.pow(expo_vals, -1.0)
        expo_vals = expo_vals.reshape(-1, 1, 1, 1)
        images = (
            torch.pow((images-self.pixel_min)/self.pixel_interval, expo_vals) *
            self.pixel_interval + self.pixel_min)
        return images, flows


class AddGaussianNoise(BaseTransform):
    """ Adds noise to the image by summing each pixel to a value sampled from
    a normal distribution.

    Args:
        stdev: float:
            The standard deviation value for the normal distribution.
    """
    def __init__(self,
                 stdev: float) -> None:
        self.stdev = stdev

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.normal(0.0, self.stdev, images.shape)
        noise = noise.type(images.dtype)
        noise = noise.to(images.device)
        images += noise
        return images, flows


class RandomScale(BaseTransform):
    """ Rescale the inputs to a new size according to the given boundaries.

    Args:
        range_min: float:
            The minimum value of the uniform distribution.
        range_max: float:
            The maximum value of the uniform distribution.
    """
    def __init__(self,
                 range_min: float,
                 range_max: float) -> None:
        self.range_min = range_min
        self.range_max = range_max
        self.interval = range_max - range_min

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.rand([1])
        scale = scale.type(flows.dtype)
        scale = scale.to(flows.device)
        scale *= self.interval
        scale += self.range_min
        new_size = (int(scale*images.shape[2]), int(scale*images.shape[3]))
        images = F.interpolate(
            images, size=new_size, mode='bilinear', align_corners=False)
        flows = F.interpolate(
            flows, size=new_size, mode='bilinear', align_corners=False)
        flows *= scale
        return images, flows


class RandomCrop(BaseTransform):
    """ Crops the inputs at a random location according to a given size.

    Args:
        size: Tuple[int, int]:
            The size [height, width] of the crop.
    """
    def __init__(self,
                 size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = images.shape
        th, tw = self.size
        if w == tw and h == th:
            return images, flows

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        images = images[:, :, y1:y1+th, x1:x1+tw]
        flows = flows[:, :, y1:y1+th, x1:x1+tw]
        return images, flows


class RandomTranslate(BaseTransform):
    """ Creates a translation between images by applying a random alternated
    crop on the sequence of inputs. A translation value t is randomly selected
    first. Then, the first image is cropped by a box translated by t. The 
    second image will be cropped by a reversed translation -t. The third will
    be cropped by t again, and so on...

    Args:
        translation: Tuple[int, int]
            The maximum absolute translation values (in pixels) in the
            vertical and horizontal axis respectively.
    """
    def __init__(self,
                 translation: Tuple[int, int]) -> None:
        self.translation = translation

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = images.shape
        th, tw = self.translation
        tw = np.random.randint(-tw, tw)
        th = np.random.randint(-th, th)
        if tw == 0 and th == 0:
            return images, flows

        new_images = images[:, :, abs(th):, abs(tw):].clone()
        new_flows = flows[:, :, abs(th):, abs(tw):].clone()

        x1, x2 = max(0, tw), min(w+tw, w)
        y1, y2 = max(0, th), min(h+th, h)
        new_images[::2] = images[::2, :, y1:y2, x1:x2]
        new_flows[::2] = flows[::2, :, y1:y2, x1:x2]
        new_flows[::2, 0] += tw
        new_flows[::2, 1] += th

        x1, x2 = max(0, -tw), min(w-tw, w)
        y1, y2 = max(0, -th), min(h-th, h)
        new_images[1::2] = images[1::2, :, y1:y2, x1:x2]
        new_flows[1::2] = flows[1::2, :, y1:y2, x1:x2]
        new_flows[1::2, 0] -= tw
        new_flows[1::2, 1] -= th

        return new_images, new_flows


class RandomRotate(BaseTransform):
    """ Applies random rotation to the inputs. The inputs are rotated around
    the center of the image. First all inputs are rotated by the same random
    major `angle`. Then, another random angle a is sampled according to `diff_angle`.
    The first image will be rotated by a. The second image will be rotated by a
    reversed angle -a. The third will be rotated by a again, and so on...

    Args:
        angle: float:
            The maximum absolute value to sample the major angle from.
        diff_angle: float: optional, default 0
            The maximum absolute value to sample the angle difference between
            consecutive images.
    """
    def __init__(self,
                 angle: float,
                 diff_angle: float = 0):
        self.angle = angle
        self.diff_angle = diff_angle

    def __call__(self,
                 images: torch.Tensor,
                 flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        major_angle = np.random.uniform(-self.angle, self.angle)
        inter_angle = np.random.uniform(-self.diff_angle, self.diff_angle)

        _, _, h, w = images.shape

        def generate_rotation_grid(rot_angle, batch_size):
            vy, vx = torch.meshgrid(torch.arange(h), torch.arange(w))
            vx = vx.type(flows.dtype)
            vy = vy.type(flows.dtype)
            vx = vx.to(flows.device)
            vy = vy.to(flows.device)
            vx -= (w-1.0)/2.0
            vy -= (h-1.0)/2.0
            angle_rad = rot_angle*2*np.pi / 360
            rotx = np.cos(angle_rad)*vx - np.sin(angle_rad)*vy
            roty = np.sin(angle_rad)*vx + np.cos(angle_rad)*vy
            rotx = rotx / ((w-1)/2)
            roty = roty / ((h-1)/2)
            rot_grid = torch.stack((rotx, roty), dim=2)[None]
            rot_grid = rot_grid.repeat(batch_size, 1, 1, 1)
            return rot_grid

        def generate_rotation_matrix(rot_angle, batch_size):
            vx, vy = torch.meshgrid(torch.arange(h), torch.arange(w))
            vx = vx.type(flows.dtype)
            vy = vy.type(flows.dtype)
            vx = vx.to(flows.device)
            vy = vy.to(flows.device)
            rotx = (vx - h/2.0) * (rot_angle*np.pi/180.0)
            roty = -(vy - w/2.0) * (rot_angle*np.pi/180.0)
            rot_mat = torch.stack((rotx, roty), dim=0)[None]
            rot_mat = rot_mat.repeat(batch_size, 1, 1, 1)
            return rot_mat

        def rotate_flow(flow, rot_angle):
            angle_rad = rot_angle*2*np.pi / 360
            rot_flow = flow.clone()
            rot_flow[:, 0] = (
                np.cos(angle_rad)*flow[:, 0] + np.sin(angle_rad)*flow[:, 1])
            rot_flow[:, 1] = (
                -np.sin(angle_rad)*flow[:, 0] + np.cos(angle_rad)*flow[:, 1])
            return rot_flow

        angle = major_angle - inter_angle / 2
        num_images = images[::2].shape[0]
        rot_grid = generate_rotation_grid(angle, num_images)
        images[::2] = F.grid_sample(images[::2], rot_grid, mode='bilinear')
        num_flows = flows[::2].shape[0]
        rot_mat = generate_rotation_matrix(inter_angle, num_flows)
        flows[::2] += rot_mat
        flows[::2] = F.grid_sample(
            flows[::2], rot_grid[:num_flows], mode='bilinear')
        flows[::2] = rotate_flow(flows[::2], angle)

        angle = major_angle + inter_angle / 2
        num_images = images[1::2].shape[0]
        rot_grid = generate_rotation_grid(angle, num_images)
        images[1::2] = F.grid_sample(images[1::2], rot_grid, mode='bilinear')
        num_flows = flows[1::2].shape[0]
        flows[1::2] -= rot_mat
        flows[1::2] = F.grid_sample(
            flows[1::2], rot_grid[:num_flows], mode='bilinear')
        flows[1::2] = rotate_flow(flows[1::2], angle)

        return images, flows