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
