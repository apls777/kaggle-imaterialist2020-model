from typing import List
import numpy as np
from skimage import measure
from itertools import groupby
from PIL import Image


def get_binary_mask(encoded_pixels: str, height: int, width: int, resize_image_size: int = None):
    """Converts RLE to a binary mask."""
    mask_len = height * width
    mask = np.zeros(mask_len, dtype=np.bool)
    encoded_pixels = [int(x) for x in encoded_pixels.split(' ')]
    for start_pixel, num_pixels in zip(encoded_pixels[::2], encoded_pixels[1::2]):
        start_pixel -= 1

        if start_pixel + num_pixels > mask_len:
            raise ValueError('wrong size')

        mask[start_pixel:start_pixel + num_pixels] = 1

    mask = mask.reshape((height, width), order='F')

    # resize the mask if necessary
    if resize_image_size is not None:
        if width > height:
            new_width = resize_image_size
            new_height = int(height / (width / resize_image_size))
        else:
            new_height = resize_image_size
            new_width = int(width / (height / resize_image_size))

        if (width != new_width) or (height != new_height):
            pil_image = Image.fromarray(mask.astype(np.uint8))
            pil_image = pil_image.resize((new_width, new_height), Image.NEAREST)
            mask = np.asarray(pil_image).astype(np.bool)

    return mask


def rle_encode(mask):
    pixels = mask.T.flatten()

    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1

    rle[1::2] = rle[1::2] - rle[:-1:2]

    return rle


def rle_decode(rle_str, mask_shape):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=np.bool)
    for lo, hi in zip(starts, ends):
        if hi > len(mask):
            raise ValueError('wrong size')

        mask[lo:hi] = 1

    return mask.reshape(mask_shape[::-1]).T


def get_bbox(mask: np.ndarray):
    """Returns a bbox for the mask in the COCO format."""
    region = measure.regionprops(mask.astype(np.uint8))[0]
    min_row, min_col, max_row, max_col = region.bbox

    bbox_width = max_col - min_col
    bbox_height = max_row - min_row

    return min_col, min_row, bbox_width, bbox_height


def convert_to_coco_rle(encoded_pixels: List[int], height: int, width: int, compressed: bool = False):
    """Converts RLE from the iMaterialist format to the COCO format."""
    prev_start_pixel = 0
    prev_num_pixels = 0
    counts = []

    for start_pixel, num_pixels in zip(encoded_pixels[::2], encoded_pixels[1::2]):
        assert start_pixel > prev_start_pixel

        coco_start_pixel = start_pixel - prev_start_pixel - prev_num_pixels - 1
        prev_start_pixel = start_pixel - 1
        prev_num_pixels = num_pixels
        counts += [coco_start_pixel, num_pixels]

    num_zeros = height * width - prev_start_pixel - prev_num_pixels
    if num_zeros:
        counts.append(num_zeros)

    rle = {'counts': counts, 'size': [height, width]}
    if compressed:
        from pycocotools import mask
        rle = mask.frPyObjects(rle, height, width)

    return rle


def binary_mask_to_rle(binary_mask: np.ndarray, compressed: bool = False):
    counts = []
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)

        counts.append(len(list(elements)))

    rle = {'counts': counts, 'size': list(binary_mask.shape)}
    if compressed:
        from pycocotools import mask
        rle = mask.frPyObjects(rle, binary_mask.shape[0], binary_mask.shape[1])

    return rle
