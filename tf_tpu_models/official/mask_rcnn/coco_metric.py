# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import atexit
import copy
import tempfile
import numpy as np

# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

from tf_tpu_models.official.mask_rcnn.coco_utils import load_predictions
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import tensorflow.compat.v1 as tf


class MaskCOCO(COCO):
  """COCO object for mask evaluation.
  """

  def reset(self, dataset):
    """Reset the dataset and groundtruth data index in this object.

    Args:
      dataset: dict of groundtruth data. It should has similar structure as the
        COCO groundtruth JSON file. Must contains three keys: {'images',
          'annotations', 'categories'}.
        'images': list of image information dictionary. Required keys: 'id',
          'width' and 'height'.
        'annotations': list of dict. Bounding boxes and segmentations related
          information. Required keys: {'id', 'image_id', 'category_id', 'bbox',
            'iscrowd', 'area', 'segmentation'}.
        'categories': list of dict of the category information.
          Required key: 'id'.
        Refer to http://cocodataset.org/#format-data for more details.

    Raises:
      AttributeError: If the dataset is empty or not a dict.
    """
    assert dataset, 'Groundtruth should not be empty.'
    assert isinstance(dataset,
                      dict), 'annotation file format {} not supported'.format(
                          type(dataset))
    self.anns, self.cats, self.imgs = dict(), dict(), dict()
    self.dataset = copy.deepcopy(dataset)
    self.createIndex()

  def loadRes(self, detection_results, include_mask, is_image_mask=False):
    """Load result file and return a result api object.

    Args:
      detection_results: a dictionary containing predictions results.
      include_mask: a boolean, whether to include mask in detection results.
      is_image_mask: a boolean, where the predict mask is a whole image mask.

    Returns:
      res: result MaskCOCO api object
    """
    res = MaskCOCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    print('Loading and preparing results...')
    _, predictions = load_predictions(
        detection_results,
        include_mask=include_mask,
        is_image_mask=is_image_mask)

    assert isinstance(predictions, list), 'results in not an array of objects'

    if predictions:
      image_ids = [pred['image_id'] for pred in predictions]
      assert set(image_ids) == (set(image_ids) & set(self.getImgIds())), \
             'Results do not correspond to current coco set'

      if (predictions and 'bbox' in predictions[0] and predictions[0]['bbox']):
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for idx, pred in enumerate(predictions):
          bb = pred['bbox']
          x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
          if 'segmentation' not in pred:
            pred['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
          pred['area'] = bb[2] * bb[3]
          pred['id'] = idx + 1
          pred['iscrowd'] = 0
      elif 'segmentation' in predictions[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for idx, pred in enumerate(predictions):
          # now only support compressed RLE format as segmentation results
          pred['area'] = maskUtils.area(pred['segmentation'])
          if 'bbox' not in pred:
            pred['bbox'] = maskUtils.toBbox(pred['segmentation'])
          pred['id'] = idx + 1
          pred['iscrowd'] = 0

      res.dataset['annotations'] = predictions

    res.createIndex()
    return res


class EvaluationMetric(object):
  """COCO evaluation metric class."""

  def __init__(self, filename, include_mask):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _evaluate() loads a JSON file in COCO annotation format as the
    groundtruths and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation.
      include_mask: boolean to indicate whether or not to include mask eval.
    """
    if filename:
      if filename.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.gfile.Remove(local_val_json)

        tf.gfile.Copy(filename, local_val_json)
        atexit.register(tf.gfile.Remove, local_val_json)
      else:
        local_val_json = filename
      self.coco_gt = MaskCOCO(local_val_json)
    self.filename = filename
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self._include_mask = include_mask
    if self._include_mask:
      mask_metric_names = ['mask_' + x for x in self.metric_names]
      self.metric_names.extend(mask_metric_names)

    self._reset()

  def _reset(self):
    """Reset COCO API object."""
    if self.filename is None and not hasattr(self, 'coco_gt'):
      self.coco_gt = MaskCOCO()

  def predict_metric_fn(self,
                        predictions,
                        is_predict_image_mask=False,
                        groundtruth_data=None):
    """Generates COCO metrics."""
    image_ids = list(set(predictions['source_id']))
    if groundtruth_data is not None:
      self.coco_gt.reset(groundtruth_data)
    coco_dt = self.coco_gt.loadRes(
        predictions, self._include_mask, is_image_mask=is_predict_image_mask)
    coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      # Create another object for instance segmentation metric evaluation.
      mcoco_eval = COCOeval(self.coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics

    # clean up after evaluation is done.
    self._reset()
    metrics = metrics.astype(np.float32)

    metrics_dict = {}
    for i, name in enumerate(self.metric_names):
      parts = name.split('_')
      group_name = 'bbox' if len(parts) == 1 else parts[0]
      metrics_dict['%s/%s' % (group_name, parts[-1])] = metrics[i]

    return metrics_dict
