import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_without_pad(
    images: np.ndarray, height: int, width: int, method: int = Image.BILINEAR
) -> np.ndarray:
    """
    Resize a single image or a (possibly nested) batch of images to
    ``height × width`` using PIL, without keeping the original aspect ratio.

    Parameters
    ----------
    images : np.ndarray
        Tensor in [..., H, W, C] memory layout.  Typical dtype is uint8.
    height, width : int
        Spatial dimensions of the output.
    method : int, optional
        PIL resampling filter (e.g. ``Image.BILINEAR``, ``Image.NEAREST``).

    Returns
    -------
    np.ndarray
        Tensor in the same leading-dimensional structure as the input but with
        the last two spatial axes replaced by (``height``, ``width``).

    Notes
    -----
    * If the incoming array already matches ``(height, width)``, it is returned
      unchanged (zero-copy).
    * PIL expects **width × height** order when calling ``resize``.
    * For non-uint8 inputs you may need to convert to/from uint8,
      depending on PIL support for your dtype and value range.
    """
    # Fast path: nothing to do
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat = images.reshape(-1, *original_shape[-3:])  # (N, H, W, C)

    def _resize_pil(im: Image.Image, h: int, w: int, m: int) -> Image.Image:
        """Resize one PIL image directly to (w, h)."""
        return im.resize((w, h), resample=m)

    resized_flat = np.stack(
        [_resize_pil(Image.fromarray(im), height, width, method) for im in flat]
    )

    # Restore the original leading dimensions
    return resized_flat.reshape(*original_shape[:-3], height, width, original_shape[-1])


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_image_tensor_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
    channels_last: bool = False,
) -> torch.Tensor:
    """PyTorch equivalent of JAX resize_with_pad function.

    Args:
        images: Tensor of shape (*batch, H, W, C) for uint8 or float32
        height: Target height
        width: Target width
        mode: Interpolation mode for F.interpolate

    Returns:
        Resized and padded images of shape (*batch, height, width, C)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images.unsqueeze(0)

    if not channels_last:
        images = images.permute(0, 2, 3, 1)
    # Get current dimensions
    cur_height, cur_width = images.shape[-3:-1]

    # Calculate resize ratio to maintain aspect ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Convert NHWC to NCHW for F.interpolate
    images_nchw = images.permute(0, 3, 1, 2)
    # For uint8 images, convert to float for interpolation to avoid precision issues
    original_dtype = images.dtype
    if original_dtype == torch.uint8:
        images_nchw = images_nchw.float()

    # Resize images
    resized_images_nchw = F.interpolate(
        images_nchw,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=True,  # Add antialiasing for better quality
    )
    # Convert back to NHWC
    resized_images = resized_images_nchw.permute(0, 2, 3, 1)
    # Handle dtype conversion back to original
    if original_dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad images - F.pad expects (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # For NHWC format, we need (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, 0, 0)
    pad_value = 0 if original_dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, 0, 0),
        mode="constant",
        value=pad_value,
    )

    if not channels_last:
        padded_images = padded_images.permute(0, 3, 1, 2)
    if not has_batch_dim:
        padded_images = padded_images.squeeze(0)

    return padded_images


def resize_image_tensor(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
    channels_last: bool = False,
) -> torch.Tensor:
    """
    Resize a single image or batch of images to (height, width) with no padding.

    Args
    ----
    images : torch.Tensor
        • 4-D NHWC or NCHW (uint8 or float32)
        • 3-D HWC or CHW → a dummy batch dimension is added internally
    height, width : int
        Spatial size of the output.
    mode : str, default "bilinear"
        Interpolation algorithm supported by ``torch.nn.functional.interpolate``.
    channels_last : bool, default False
        If True, the tensor is assumed NHWC; otherwise NCHW.

    Returns
    -------
    torch.Tensor
        Resized tensor with shape matching the input layout
        (batch? , height , width , channels) or (batch? , channels , height , width).

    Notes
    -----
    * ``uint8`` inputs are cast to ``float32`` for interpolation and cast back,
      ensuring sub-pixel accuracy without overflow.
    * Float inputs are clamped to ``[-1, 1]``—matching common normalized image ranges.
    """
    had_batch = images.ndim == 4
    if not had_batch:  # add batch dim if a single image
        images = images.unsqueeze(0)

    # Convert to NHWC to make spatial dims contiguous regardless of layout
    if not channels_last:
        images = images.permute(0, 2, 3, 1)

    # NHWC → NCHW for F.interpolate
    images_nchw = images.permute(0, 3, 1, 2)
    orig_dtype = images_nchw.dtype
    if orig_dtype == torch.uint8:
        images_nchw = images_nchw.float()

    # --- core resize (no padding) -------------------------------------------
    resized = F.interpolate(
        images_nchw,
        size=(height, width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=True,
    )

    # Restore dtype / range
    if orig_dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    else:  # assume float32/16 in [-1, 1] or [0, 1] and just clamp conservatively
        resized = resized.clamp(-1.0, 1.0)

    # Back to original layout
    resized = resized.permute(0, 2, 3, 1)  # NCHW → NHWC
    if not channels_last:
        resized = resized.permute(0, 3, 1, 2)  # NHWC → NCHW

    if not had_batch:
        resized = resized.squeeze(0)

    return resized
