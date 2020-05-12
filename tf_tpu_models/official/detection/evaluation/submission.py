import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary_iterator
import os


def encode_mask(mask: np.ndarray) -> str:
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

    return ' '.join(str(x) for x in rle)


def get_metrics(model_dir: str, step: int):
    """Returns the best evaluation result based on the compare function."""
    eval_result = {}
    for event_file in gfile.Glob(os.path.join(model_dir, 'eval', '*.tfevents.*')):
        for event in summary_iterator.summary_iterator(event_file):
            if event.step == step:
                assert event.HasField('summary')

                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        eval_result[value.tag] = value.simple_value

                break

    return eval_result


def get_new_image_size(image_size, output_size: int):
    image_height, image_width = image_size

    if image_width > image_height:
        scale = image_width / output_size
        new_width = output_size
        new_height = int(image_height / scale)
    else:
        scale = image_height / output_size
        new_height = output_size
        new_width = int(image_width / scale)

    return new_height, new_width
