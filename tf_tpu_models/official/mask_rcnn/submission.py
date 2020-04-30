import json
import logging
from tf_tpu_models.official.mask_rcnn.coco_utils import generate_segmentation_from_masks
from tf_tpu_models.official.mask_rcnn.evaluation import process_prediction_for_eval
import six
import numpy as np
from PIL import Image
import os
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary_iterator


SUBMISSION_IMAGE_SIZE = 1024


def generate_submission(eval_estimator,
                        input_fn,
                        checkpoint_path,
                        num_attributes):
    """Runs COCO evaluation once."""
    predictor = eval_estimator.predict(input_fn=input_fn, yield_single_examples=False, checkpoint_path=checkpoint_path)
    # Every predictor.next() gets a batch of prediction (a dictionary).

    # get attribute thresholds
    step = int(checkpoint_path.rsplit('-', 1)[-1])
    attr_thresholds = get_attribute_thresholds(eval_estimator.model_dir, step, num_attributes)

    # load image IDs
    with open('/workspace/project/data/test_coco.json') as f:
        test_annotations = json.load(f)

    image_filenames = {int(image['id']): image['file_name'] for image in test_annotations['images']}

    batch_idx = 0
    rows = []
    while True:
        try:
            batch_predictions = six.next(predictor)
            logging.info('Running inference on batch %d...', (batch_idx + 1))
        except StopIteration:
            logging.info('Finished the eval set at %d batch.', (batch_idx + 1))
            break

        batch_predictions = process_prediction_for_eval(batch_predictions)
        rows += _generate_submission_rows(batch_predictions, attr_thresholds, image_filenames)
        batch_idx += 1

    return rows


def _generate_submission_rows(predictions, attr_thresholds, image_filenames):
    rows = []
    for i, image_id in enumerate(predictions['source_id']):
        if (i + 1) % 100 == 0:
            logging.info('  loading image %d/%d...' % (i + 1, len(predictions['source_id'])))

        image_height = int(predictions['image_info'][i][3])
        image_width = int(predictions['image_info'][i][4])

        if image_width > image_height:
            new_width = SUBMISSION_IMAGE_SIZE
            new_height = int(image_height / (image_width / new_width))
        else:
            new_height = SUBMISSION_IMAGE_SIZE
            new_width = int(image_width / (image_height / new_height))

        for box_index in range(int(predictions['num_detections'][i])):
            mask = generate_segmentation_from_masks(predictions['detection_masks'][i][box_index:(box_index + 1)],
                                                    predictions['detection_boxes'][i][box_index:(box_index + 1)],
                                                    image_height,
                                                    image_width,
                                                    is_image_mask=False)[0]

            pil_image = Image.fromarray(mask.astype(np.uint8))
            pil_image = pil_image.resize((new_width, new_height), Image.NEAREST)
            resized_binary_mask = np.asarray(pil_image)
            encoded_mask = rle_encode(resized_binary_mask)

            # get attributes
            attr_predictions = predictions['detection_attributes'][i][box_index]
            attr_ids = np.argwhere(attr_predictions >= attr_thresholds).flatten()

            bbox_x, bbox_y, bbox_w, bbox_h = predictions['detection_boxes'][i][box_index]

            row = {
                'ImageId': image_filenames[int(image_id)].split('.')[0],
                'EncodedPixels': ' '.join(str(x) for x in encoded_mask),
                'ClassId': int(predictions['detection_classes'][i][box_index]) - 1,
                'AttributesIds': ','.join(str(attr_id) for attr_id in attr_ids),
                'image_width': new_width,
                'image_height': new_height,
                'mask_area': resized_binary_mask.sum(),
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_width': bbox_w,
                'bbox_height': bbox_h,
                'score': predictions['detection_scores'][i][box_index],
            }

            rows.append(row)

    return rows


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


def get_attribute_thresholds(model_dir: str, step: int, num_attributes: int):
    """Returns the best evaluation result based on the compare function."""
    eval_result = None
    for event_file in gfile.Glob(os.path.join(model_dir, 'eval', '*.tfevents.*')):
        for event in summary_iterator.summary_iterator(event_file):
            if event.step == step:
                assert event.HasField('summary')

                eval_result = {}
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        eval_result[value.tag] = value.simple_value

                break

    thresholds = np.zeros(num_attributes, dtype=np.float32)
    for metric_name, value in eval_result.items():
        if metric_name.startswith('attribute_threshold/attr_'):
            attr_id = int(metric_name.rsplit('_', 1)[-1])
            thresholds[attr_id] = value

    return thresholds
