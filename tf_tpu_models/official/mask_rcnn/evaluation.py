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
"""Functions to perform COCO evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from collections import defaultdict
from absl import logging
import numpy as np
from PIL import Image
import six
import tensorflow.compat.v1 as tf
from pycocotools import mask as mask_utils
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from . import coco_metric
from . import coco_utils
from .coco_metric import load_predictions
from .object_detection import visualization_utils


def process_prediction_for_eval(prediction):
  """Process the model prediction for COCO eval."""
  image_info = prediction['image_info']
  box_coordinates = prediction['detection_boxes']
  processed_box_coordinates = np.zeros_like(box_coordinates)

  for image_id in range(box_coordinates.shape[0]):
    # if evaluation uses an annotation file, it always will be done on the original image
    scale = prediction['eval_scale'][image_id] if 'eval_scale' in prediction else image_info[image_id][2]
    for box_id in range(box_coordinates.shape[1]):
      # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
      # by image scale.
      y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
      new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
      processed_box_coordinates[image_id, box_id, :] = new_box
  prediction['detection_boxes'] = processed_box_coordinates
  return prediction


def compute_coco_eval_metric(predictions,
                             include_mask=True,
                             annotation_json_file=None):
  """Compute COCO eval metric given a prediction generator.

  Args:
    predictor: a generator that iteratively pops a dictionary of predictions
      with the format compatible with COCO eval tool.
    num_batches: the number of batches to be aggregated in eval. This is how
      many times that the predictor gets pulled.
    include_mask: a boolean that indicates whether we include the mask eval.
    annotation_json_file: the annotation json file of the eval dataset.

  Returns:
    eval_results: the aggregated COCO metric eval results.
  """
  if not annotation_json_file:
    annotation_json_file = None
  use_groundtruth_from_json = (annotation_json_file is not None)

  if use_groundtruth_from_json:
    eval_metric = coco_metric.EvaluationMetric(
        annotation_json_file, include_mask=include_mask)
    eval_results = eval_metric.predict_metric_fn(predictions)
  else:
    images, annotations = coco_utils.extract_coco_groundtruth(
        predictions, include_mask)
    dataset = coco_utils.create_coco_format_dataset(images, annotations)
    eval_metric = coco_metric.EvaluationMetric(
        filename=None, include_mask=include_mask)
    eval_results = eval_metric.predict_metric_fn(
        predictions, groundtruth_data=dataset)

  logging.info('Eval results: %s', eval_results)

  return eval_results


def evaluate(eval_estimator,
             input_fn,
             num_eval_samples,
             eval_batch_size,
             include_mask=True,
             validation_json_file=None):
  """Runs COCO evaluation once."""
  predictor = eval_estimator.predict(
      input_fn=input_fn, yield_single_examples=False)
  # Every predictor.next() gets a batch of prediction (a dictionary).

  num_eval_times = num_eval_samples // eval_batch_size
  assert num_eval_times > 0, 'num_eval_samples >= eval_batch_size!'

  predictions = _get_predictions(predictor)
  eval_results = compute_coco_eval_metric(predictions, include_mask, validation_json_file)

  attr_eval_results = _evaluate_attributes(predictions)
  eval_results.update(attr_eval_results)

  return eval_results, predictions


def _evaluate_attributes(predictions: dict):
    _, annotations_pred = load_predictions(predictions, include_mask=False, include_attributes=True)
    _, annotations_gt = coco_utils.extract_coco_groundtruth(predictions, include_mask=False, include_attributes=True)

    # get a number of attributes
    num_attributes = len(annotations_gt[0]['attributes_multi_hot'])

    # extract category IDs
    category_ids = sorted(set(int(ann['category_id']) for ann in annotations_gt))

    # group annotations and predictions by image ID and category ID
    annotations_pred = _group_annotations(annotations_pred)
    annotations_gt = _group_annotations(annotations_gt)

    # get image IDs
    image_ids = list(annotations_gt.keys())

    # collect all attribute predictions and labels
    attributes_pred = []
    attributes_gt = []
    for category_id in category_ids:
        for image_id in image_ids:
            boxes_pred = annotations_pred[image_id].get(category_id)
            boxes_gt = annotations_gt[image_id].get(category_id)
            if not boxes_pred or not boxes_gt:
                continue

            pred_bboxes = [box['bbox'] for box in boxes_pred]
            gt_bboxes = [box['bbox'] for box in boxes_gt]
            is_crowd = [0] * len(gt_bboxes)
            ious = mask_utils.iou(pred_bboxes, gt_bboxes, is_crowd)

            for i, box_pred in enumerate(boxes_pred):
                for j, box_gt in enumerate(boxes_gt):
                    if ious[i, j] > 0.5:
                        attributes_pred.append(box_pred['attributes_multi_hot'])
                        attributes_gt.append(box_gt['attributes_multi_hot'])

    attributes_pred = np.array(attributes_pred)
    attributes_gt = np.array(attributes_gt)

    # compute attribute thresholds and metrics
    attr_precisions = np.zeros(num_attributes, dtype=np.float32)
    attr_recalls = np.zeros(num_attributes, dtype=np.float32)
    attr_f1_scores = np.zeros(num_attributes, dtype=np.float32)
    attr_thresholds = np.zeros(num_attributes, dtype=np.float32)

    for attr_id in range(num_attributes):
        precisions, recalls, thresholds = precision_recall_curve(attributes_gt[:, attr_id], attributes_pred[:, attr_id])

        best_precision = 0
        best_recall = 0
        best_f1_score = 0
        best_threshold = 0
        for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
            f1_score_val = compute_f1_score(precision, recall)
            if f1_score_val > best_f1_score:
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score_val
                best_threshold = threshold

        attr_precisions[attr_id] = best_precision
        attr_recalls[attr_id] = best_recall
        attr_f1_scores[attr_id] = best_f1_score
        attr_thresholds[attr_id] = best_threshold

    # evaluate attribute predictions for boxes
    all_cat_precisions = {}
    all_cat_recalls = {}
    all_cat_f1_scores = {}
    for category_id in category_ids:
        cat_precisions = []
        cat_recalls = []
        cat_f1_scores = []

        for image_id in image_ids:
            boxes_pred = annotations_pred[image_id][category_id]
            boxes_gt = annotations_gt[image_id][category_id]

            pred_bboxes = [box['bbox'] for box in boxes_pred]
            gt_bboxes = [box['bbox'] for box in boxes_gt]
            is_crowd = [0] * len(gt_bboxes)
            ious = mask_utils.iou(pred_bboxes, gt_bboxes, is_crowd)

            img_precisions = []
            img_recalls = []
            img_f1_scores = []
            for i, box_pred in enumerate(boxes_pred):
                for j, box_gt in enumerate(boxes_gt):
                # if sum(box_gt['attributes_multi_hot']):
                    if ious[i, j] > 0.5:
                        attr_pred = (box_pred['attributes_multi_hot'] >= attr_thresholds).astype(np.int32)
                        attr_gt = box_gt['attributes_multi_hot']
                        # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

                        has_pred = (attr_pred > 0).any()
                        has_gt = (attr_gt > 0).any()

                        if (has_pred and not has_gt) or (not has_pred and has_gt):
                            precision, recall, f1_score = 0., 0., 0.
                        elif not has_gt and not has_pred:
                            precision, recall, f1_score = 1., 1., 1.
                        else:
                            precision, recall, f1_score, _ = precision_recall_fscore_support(attr_gt, attr_pred, average='binary')

                        img_precisions.append(precision)
                        img_recalls.append(recall)
                        img_f1_scores.append(f1_score)

            if img_precisions:
                cat_precisions.append(sum(img_precisions) / len(img_precisions))
                cat_recalls.append(sum(img_recalls) / len(img_recalls))
                cat_f1_scores.append(sum(img_f1_scores) / len(img_f1_scores))

        # average attribute precision, recall and F1-score across all images
        all_cat_precisions[category_id] =  (sum(cat_precisions) / len(cat_precisions)) if cat_precisions else None
        all_cat_recalls[category_id] = (sum(cat_recalls) / len(cat_recalls)) if cat_recalls else None
        all_cat_f1_scores[category_id] = (sum(cat_f1_scores) / len(cat_f1_scores)) if cat_f1_scores else None

    eval_results = {}
    for attr_id in range(num_attributes):
        eval_results['attribute_precision/attr_%03d' % attr_id] = attr_precisions[attr_id]
        eval_results['attribute_recall/attr_%03d' % attr_id] = attr_recalls[attr_id]
        eval_results['attribute_f1_score/attr_%03d' % attr_id] = attr_f1_scores[attr_id]
        eval_results['attribute_threshold/attr_%03d' % attr_id] = attr_thresholds[attr_id]

    for category_id in category_ids:
        eval_results['category_attributes_precision/cat_%02d' % category_id] = all_cat_precisions[category_id]
        eval_results['category_attributes_recall/cat_%02d' % category_id] = all_cat_recalls[category_id]
        eval_results['category_attributes_f1_score/cat_%02d' % category_id] = all_cat_f1_scores[category_id]

    return eval_results


def _group_annotations(annotations: list):
    grouped_annotations = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        grouped_annotations[int(ann['image_id'])][int(ann['category_id'])].append(ann)

    return grouped_annotations


def compute_f1_score(precision: float, recall: float):
    if precision + recall == 0.:
        return 0.

    return (2 * precision * recall) / (precision + recall)


def _get_predictions(predictor):
    batch_idx = 0
    predictions = dict()
    while True:
        try:
            prediction = six.next(predictor)
            logging.info('Running inference on batch %d...', (batch_idx + 1))
        except StopIteration:
            logging.info('Finished the eval set at %d batch.', (batch_idx + 1))
            break

        prediction = process_prediction_for_eval(prediction)

        for k, v in six.iteritems(prediction):
            if k not in predictions:
                predictions[k] = [v]
            else:
                predictions[k].append(v)

        batch_idx = batch_idx + 1

    for k, v in six.iteritems(predictions):
        predictions[k] = np.concatenate(predictions[k], axis=0)

    if 'orig_images' in predictions and predictions['orig_images'].shape[0] > 10:
        # Only samples a few images for visualization.
        predictions['orig_images'] = predictions['orig_images'][:10]

    return predictions


def write_summary(eval_results, summary_writer, current_step, predictions=None):
  """Write out eval results for the checkpoint."""
  with tf.Graph().as_default():
    summaries = []
    for metric in eval_results:
      summaries.append(
          tf.Summary.Value(tag=metric, simple_value=eval_results[metric]))
    tf_summary = tf.Summary(value=list(summaries))
    summary_writer.add_summary(tf_summary, current_step)
  write_image_summary(predictions, summary_writer, current_step)


def create_image_summary(image,
                         boxes,
                         scores,
                         classes,
                         gt_boxes=None,
                         segmentations=None):
  """Creates an image summary given predictions."""
  max_boxes_to_draw = 100
  min_score_thresh = 0.1

  # Visualizes the predicitons.
  image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes=classes,
      scores=scores,
      category_index={},
      instance_masks=segmentations,
      use_normalized_coordinates=False,
      max_boxes_to_draw=max_boxes_to_draw,
      min_score_thresh=min_score_thresh,
      agnostic_mode=False)
  if gt_boxes is not None:
    # Visualizes the groundtruth boxes. They are in black by default.
    image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        gt_boxes,
        classes=None,
        scores=None,
        category_index={},
        use_normalized_coordinates=False,
        max_boxes_to_draw=max_boxes_to_draw,
        agnostic_mode=True)
  buf = io.BytesIO()
  w, h = image_with_detections.shape[:2]
  ratio = 1024 / w
  new_size = [int(w * ratio), int(h * ratio)]
  image = Image.fromarray(image_with_detections.astype(np.uint8))
  image.thumbnail(new_size)
  image.save(buf, format='png')
  image_summary = tf.Summary.Image(encoded_image_string=buf.getvalue())
  return image_summary


def write_image_summary(predictions, summary_writer, current_step):
  """Write out image and prediction for summary."""
  if not predictions or not isinstance(predictions, dict):
    return
  if 'orig_images' not in predictions:
    logging.info('Missing orig_images in predictions: %s',
                 predictions.keys())
    return
  predictions['orig_images'] = predictions['orig_images'] * 255
  predictions['orig_images'] = predictions['orig_images'].astype(np.uint8)
  num_images = predictions['orig_images'].shape[0]
  include_mask = ('detection_masks' in predictions)

  with tf.Graph().as_default():
    summaries = []
    for i in xrange(num_images):
      num_detections = min(
          len(predictions['detection_boxes'][i]),
          int(predictions['num_detections'][i]))
      detection_boxes = predictions['detection_boxes'][i][:num_detections]
      detection_scores = predictions['detection_scores'][i][:num_detections]
      detection_classes = predictions['detection_classes'][i][:num_detections]

      image = predictions['orig_images'][i]
      image_height = image.shape[0]
      image_width = image.shape[1]

      # Rescale the box to fit the visualization image.
      h, w = predictions['image_info'][i][3:5]
      detection_boxes = detection_boxes / np.array([w, h, w, h])
      detection_boxes = detection_boxes * np.array(
          [image_width, image_height, image_width, image_height])

      gt_boxes = None
      if 'groundtruth_boxes' in predictions:
        gt_boxes = predictions['groundtruth_boxes'][i]
        gt_boxes = gt_boxes * np.array(
            [image_height, image_width, image_height, image_width])

      segmentations = None
      if include_mask:
        instance_masks = predictions['detection_masks'][i][0:num_detections]
        segmentations = coco_metric.generate_segmentation_from_masks(
            instance_masks, detection_boxes, image_height, image_width)

      # From [x, y, w, h] to [x1, y1, x2, y2] and
      # process_prediction_for_eval() set the box to be [x, y] format, need to
      # reverted them to [y, x] format.
      xmin, ymin, w, h = np.split(detection_boxes, 4, axis=-1)
      xmax = xmin + w
      ymax = ymin + h
      boxes_to_visualize = np.concatenate([ymin, xmin, ymax, xmax], axis=-1)
      image_summary = create_image_summary(
          image,
          boxes=boxes_to_visualize,
          scores=detection_scores,
          classes=detection_classes.astype(np.int32),
          gt_boxes=gt_boxes,
          segmentations=segmentations)
      image_value = tf.Summary.Value(tag='%d_input' % i, image=image_summary)
      summaries.append(image_value)
    tf_summary = tf.Summary(value=list(summaries))
    summary_writer.add_summary(tf_summary, current_step)
