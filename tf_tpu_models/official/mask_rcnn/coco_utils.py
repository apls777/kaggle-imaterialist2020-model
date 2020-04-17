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

import numpy as np
import pycocotools.mask as maskUtils
from tf_tpu_models.official.mask_rcnn.coco_metric import generate_segmentation_from_masks


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


def extract_coco_groundtruth(prediction, include_mask=False):
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
