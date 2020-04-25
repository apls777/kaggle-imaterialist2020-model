# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Util functions to manipulate masks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import pycocotools.mask as maskUtils
import cv2


POLYGON_PAD_VALUE = -3
POLYGON_SEPARATOR = -1
MASK_SEPARATOR = -2


def _np_array_split(a, v):
  """Split numpy array by separator value.

  Args:
    a: 1-D numpy.array.
    v: number. Separator value. e.g -1.

  Returns:
    2-D list of clean separated arrays.

  Example:
    a = [1, 2, 3, 4, -1, 5, 6, 7, 8]
    b = _np_array_split(a, -1)
    # Output: b = [[1, 2, 3, 4], [5, 6, 7, 8]]
  """
  a = np.array(a)
  arrs = np.split(a, np.where(a[:] == v)[0])
  return [e if (len(e) <= 0 or e[0] != v) else e[1:] for e in arrs]


def _unflat_polygons(x):
  """Unflats/recovers 1-d padded polygons to 3-d polygon list.

  Args:
    x: numpay.array. shape [num_elements, 1], num_elements = num_obj *
      num_vertex + padding.

  Returns:
    A list of three dimensions: [#obj, #polygon, #vertex]
  """
  num_segs = _np_array_split(x, MASK_SEPARATOR)
  polygons = []
  for s in num_segs:
    polygons.append(_np_array_split(s, POLYGON_SEPARATOR))
  polygons = [[polygon.tolist() for polygon in obj] for obj in polygons]  # pylint: disable=g-complex-comprehension
  return polygons


def _denormalize_to_coco_bbox(bbox, height, width):
  """Denormalize bounding box.

  Args:
    bbox: numpy.array[float]. Normalized bounding box. Format: ['ymin', 'xmin',
      'ymax', 'xmax'].
    height: int. image height.
    width: int. image width.

  Returns:
    [x, y, width, height]
  """
  y1, x1, y2, x2 = bbox
  y1 *= height
  x1 *= width
  y2 *= height
  x2 *= width
  box_height = y2 - y1
  box_width = x2 - x1
  return [float(x1), float(y1), float(box_width), float(box_height)]


def _extract_image_info(prediction, b):
  return {
      'id': int(prediction['source_id'][b]),
      'width': int(prediction['eval_width'][b]),
      'height': int(prediction['eval_height'][b]),
  }


def _extract_bbox_annotation(prediction, b, obj_i):
  """Constructs COCO format bounding box annotation."""
  height = prediction['eval_height'][b]
  width = prediction['eval_width'][b]
  bbox = _denormalize_to_coco_bbox(
      prediction['groundtruth_boxes'][b][obj_i, :], height, width)
  if 'groundtruth_area' in prediction:
    area = float(prediction['groundtruth_area'][b][obj_i])
  else:
    # Using the box area to replace the polygon area. This value will not affect
    # real evaluation but may fail the unit test.
    area = bbox[2] * bbox[3]

  annotation = {
      'id': b * 1000 + obj_i,  # place holder of annotation id.
      'image_id': int(prediction['source_id'][b]),  # source_id,
      'category_id': int(prediction['groundtruth_classes'][b][obj_i]),
      'bbox': bbox,
      'iscrowd': int(prediction['groundtruth_is_crowd'][b][obj_i]),
      'area': area,
      'segmentation': [],
  }

  return annotation


def _extract_segmentaton_info(prediction, bbox, b, obj_i):
  """Constructs 'area' and 'segmentation' fields.

  Args:
    prediction: dict[str, numpy.array]. Model outputs. The value dimension is
      [batch_size, #objects, #features, ...]
    polygons: list[list[list]]. Dimensions are [#objects, #polygon, #vertex].
    b: batch index.
    obj_i: object index.

  Returns:
    dict[str, numpy.array]. COCO format annotation with 'area' and
    'segmentation'.
  """
  annotation = {
    'area': float(prediction['groundtruth_area'][b][obj_i]),
  }

  height = prediction['eval_height'][b]
  width = prediction['eval_width'][b]
  instance_mask = generate_segmentation_from_masks(
    np.expand_dims(prediction['groundtruth_cropped_masks'][b][obj_i].astype(np.float32), axis=0),
    np.expand_dims(bbox, axis=0),
    int(height),
    int(width),
    is_image_mask=False,
  )[0]

  # Convert the mask to uint8 and then to fortranarray for RLE encoder.
  annotation['segmentation'] = maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
  annotation['segmentation']['counts'] = annotation['segmentation']['counts'].decode('utf-8')

  return annotation


def _extract_categories(annotations):
  """Extract categories from annotations."""
  categories = {}
  for anno in annotations:
    category_id = int(anno['category_id'])
    categories[category_id] = {'id': category_id}
  return list(categories.values())


def extract_coco_groundtruth(prediction, include_mask=False, include_attributes=False):
  """Extract COCO format groundtruth.

  Args:
    prediction: dictionary of batch of prediction result. the first dimension
      each element is the batch.
    include_mask: True for including masks in the output annotations.

  Returns:
    Tuple of (images, annotations).
    images: list[dict].Required keys: 'id', 'width' and 'height'. The values are
      image id, width and height.
    annotations: list[dict]. Required keys: {'id', 'source_id', 'category_id',
      'bbox', 'iscrowd'} when include_mask=False. If include_mask=True, also
      required {'area', 'segmentation'}. The 'id' value is the annotation id
      and can be any **positive** number (>=1).
      Refer to http://cocodataset.org/#format-data for more details.
  Raises:
    ValueError: If any groundtruth fields is missing.
  """
  required_fields = [
      'source_id', 'eval_width', 'eval_height', 'num_groundtruth_labels',
      'groundtruth_boxes', 'groundtruth_classes'
  ]
  if include_mask:
    required_fields += ['groundtruth_area', 'groundtruth_cropped_masks']

  for key in required_fields:
    if key not in prediction.keys():
      raise ValueError('Missing groundtruth field: "{}" keys: {}'.format(
          key, prediction.keys()))

  print('Extracting GT data for evaluation...')

  images = []
  annotations = []
  for b in range(prediction['source_id'].shape[0]):
    if (b + 1) % 100 == 0:
      print('  image %d/%d' % (b + 1, prediction['source_id'].shape[0]))

    # Constructs image info.
    image = _extract_image_info(prediction, b)
    images.append(image)

    # Constructs annotations.
    num_labels = prediction['num_groundtruth_labels'][b]
    for obj_i in range(num_labels):
      annotation = _extract_bbox_annotation(prediction, b, obj_i)

      if include_mask:
        segmentation_info = _extract_segmentaton_info(prediction, annotation['bbox'], b, obj_i)
        annotation.update(segmentation_info)

      if include_attributes:
        annotation['attributes_multi_hot'] = prediction['groundtruth_attributes'][b][obj_i]

      annotations.append(annotation)

  return images, annotations


def create_coco_format_dataset(images,
                               annotations,
                               regenerate_annotation_id=True):
  """Creates COCO format dataset with COCO format images and annotations."""
  if regenerate_annotation_id:
    for i in range(len(annotations)):
      # WARNING: The annotation id must be positive.
      annotations[i]['id'] = i + 1

  categories = _extract_categories(annotations)
  dataset = {
      'images': images,
      'annotations': annotations,
      'categories': categories,
  }
  return dataset


def load_predictions(detection_results,
                     include_mask=False,
                     is_image_mask=False,
                     include_attributes=False):
  """Create prediction dictionary list from detection and mask results.

  Args:
    detection_results: a dictionary containing numpy arrays which corresponds
      to prediction results.
    include_mask: a boolean, whether to include mask in detection results.
    is_image_mask: a boolean, where the predict mask is a whole image mask.

  Returns:
    a list of dictionary including different prediction results from the model
      in numpy form.
  """
  images = []
  predictions = []
  for i, image_id in enumerate(detection_results['source_id']):
    if (i + 1) % 100 == 0:
      logging.info('  loading image %d/%d...' % (i + 1, len(detection_results['source_id'])))

    image_height = detection_results['eval_height'][i] if 'eval_height' in detection_results \
      else detection_results['image_info'][i][3]
    image_width = detection_results['eval_width'][i] if 'eval_width' in detection_results \
      else detection_results['image_info'][i][4]

    images.append({
      'id': int(image_id),
      'width': int(image_width),
      'height': int(image_height),
    })

    if include_mask:
      box_coorindates_in_image = detection_results['detection_boxes'][i]
      segments = generate_segmentation_from_masks(
          detection_results['detection_masks'][i],
          box_coorindates_in_image,
          int(image_height),
          int(image_width),
          is_image_mask=is_image_mask)

      # Convert the mask to uint8 and then to fortranarray for RLE encoder.
      encoded_masks = [
          maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
          for instance_mask in segments
      ]

    for box_index in range(int(detection_results['num_detections'][i])):
      prediction = {
          'image_id': int(image_id),
          'bbox': detection_results['detection_boxes'][i][box_index].tolist(),
          'score': detection_results['detection_scores'][i][box_index],
          'category_id': int(detection_results['detection_classes'][i][box_index]),
      }

      if include_attributes:
        prediction['attributes_multi_hot'] = detection_results['detection_attributes'][i][box_index]

      if include_mask:
        prediction['segmentation'] = encoded_masks[box_index]

      predictions.append(prediction)

  return images, predictions


def generate_segmentation_from_masks(masks,
                                     detected_boxes,
                                     image_height,
                                     image_width,
                                     is_image_mask=False):
  """Generates segmentation result from instance masks.

  Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.
    is_image_mask: bool. True: input masks are whole-image masks. False: input
      masks are bounding-box level masks.

  Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
  """

  def expand_boxes(boxes, scale):
    """Expands an array of boxes by a given scale."""
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
    # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
    # whereas `boxes` here is in [x1, y1, w, h] form
    w_half = boxes[:, 2] * .5
    h_half = boxes[:, 3] * .5
    x_c = boxes[:, 0] + w_half
    y_c = boxes[:, 1] + h_half

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

  # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
  # To work around an issue with cv2.resize (it seems to automatically pad
  # with repeated border values), we manually zero-pad the masks by 1 pixel
  # prior to resizing back to the original image resolution. This prevents
  # "top hat" artifacts. We therefore need to expand the reference boxes by an
  # appropriate factor.
  _, mask_height, mask_width = masks.shape
  scale = max((mask_width + 2.0) / mask_width,
              (mask_height + 2.0) / mask_height)

  ref_boxes = expand_boxes(detected_boxes, scale)
  ref_boxes = ref_boxes.astype(np.int32)
  padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
  segms = []
  for mask_ind, mask in enumerate(masks):
    im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if is_image_mask:
      # Process whole-image masks.
      im_mask[:, :] = mask[:, :]
    else:
      # Process mask inside bounding boxes.
      padded_mask[1:-1, 1:-1] = mask[:, :]

      ref_box = ref_boxes[mask_ind, :]
      w = ref_box[2] - ref_box[0] + 1
      h = ref_box[3] - ref_box[1] + 1
      w = np.maximum(w, 1)
      h = np.maximum(h, 1)

      mask = cv2.resize(padded_mask, (w, h))
      mask = np.array(mask > 0.5, dtype=np.uint8)

      x_0 = max(ref_box[0], 0)
      x_1 = min(ref_box[2] + 1, image_width)
      y_0 = max(ref_box[1], 0)
      y_1 = min(ref_box[3] + 1, image_height)

      im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (
          x_0 - ref_box[0]):(x_1 - ref_box[0])]
    segms.append(im_mask)

  segms = np.array(segms)
  assert masks.shape[0] == segms.shape[0]

  return segms
