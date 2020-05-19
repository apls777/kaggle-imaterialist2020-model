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
"""Data parser and processing for Mask R-CNN."""

import tensorflow.compat.v1 as tf
from absl import logging
from dataloader import anchor
from dataloader import mode_keys as ModeKeys
from dataloader import tf_example_decoder
from utils import box_utils
from utils import dataloader_utils
from utils import input_utils


class Parser(object):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               rpn_match_threshold=0.7,
               rpn_unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               use_autoaugment=False,
               autoaugment_policy_name='v0',
               apply_autoaugment_after_resizing=False,
               skip_crowd_during_training=True,
               max_num_instances=100,
               include_mask=False,
               mask_crop_size=112,
               use_bfloat16=True,
               mode=None,
               num_attributes=None):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      rpn_match_threshold:
      rpn_unmatched_threshold:
      rpn_batch_size_per_im:
      rpn_fg_fraction:
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      use_autoaugment: `bool`, if True, use the AutoAugment augmentation policy
        during training.
      autoaugment_policy_name: `string` that specifies the name of the
        AutoAugment policy that will be used during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      include_mask: a bool to indicate whether parse mask groundtruth.
      mask_crop_size: the size which groundtruth mask is cropped to.
      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction
        or prediction with groundtruths in the outputs.
    """
    self._mode = mode
    self._max_num_instances = max_num_instances
    self._skip_crowd_during_training = skip_crowd_during_training
    self._is_training = (mode == ModeKeys.TRAIN)

    self._example_decoder = tf_example_decoder.TfExampleDecoder(
        include_mask=include_mask,
        num_attributes=num_attributes,
    )

    # Anchor.
    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level
    self._num_scales = num_scales
    self._aspect_ratios = aspect_ratios
    self._anchor_size = anchor_size

    # Target assigning.
    self._rpn_match_threshold = rpn_match_threshold
    self._rpn_unmatched_threshold = rpn_unmatched_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # Data Augmentation with AutoAugment.
    self._use_autoaugment = use_autoaugment
    self._apply_autoaugment_after_resizing = apply_autoaugment_after_resizing
    self._autoaugment_policy_name = autoaugment_policy_name

    # Mask.
    self._include_mask = include_mask
    self._mask_crop_size = mask_crop_size

    # Attributes.
    self._num_attributes = num_attributes

    # Device.
    self._use_bfloat16 = use_bfloat16

    # Data is parsed depending on the model Modekey.
    if mode == ModeKeys.TRAIN:
      if self._use_autoaugment and self._apply_autoaugment_after_resizing:
        self._parse_fn = self._parse_train_data_v2
      else:
        self._parse_fn = self._parse_train_data
    elif mode == ModeKeys.EVAL:
      self._parse_fn = self._parse_eval_data
    elif mode == ModeKeys.PREDICT or mode == ModeKeys.PREDICT_WITH_GT:
      self._parse_fn = self._parse_predict_data
    else:
      raise ValueError('mode is not defined.')

  def __call__(self, value):
    """Parses data to an image and associated training labels.

    Args:
      value: a string tensor holding a serialized tf.Example proto.

    Returns:
      image, labels: if mode == ModeKeys.TRAIN. see _parse_train_data.
      {'images': image, 'labels': labels}: if mode == ModeKeys.PREDICT
        or ModeKeys.PREDICT_WITH_GT.
    """
    with tf.name_scope('parser'):
      data = self._example_decoder.decode(value, self._max_num_instances)
      return self._parse_fn(data)

  def _parse_train_data(self, data):
    """Parses data for training.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        rpn_score_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location]. The height_l and
          width_l represent the dimension of class logits at l-th level.
        rpn_box_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        gt_boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        gt_classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        gt_masks: groundtrugh masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
    """
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']

    if self._include_mask:
      masks = data['groundtruth_instance_masks']

    if self._num_attributes:
      attributes = data['groundtruth_attributes']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training and self._is_training:
      num_groundtrtuhs = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)

      if self._include_mask:
        masks = tf.gather(masks, indices)

      if self._num_attributes:
        attributes = tf.gather(attributes, indices)

    # Gets original image and its size.
    image = data['image']

    # NOTE: The autoaugment method works best when used alongside the standard
    # horizontal flipping of images along with size jittering and normalization.
    if self._use_autoaugment:
      try:
        from utils import autoaugment_utils  # pylint: disable=g-import-not-at-top
      except ImportError as e:
        logging.exception('Autoaugment is not supported in TF 2.x.')
        raise e

      image, boxes, masks = autoaugment_utils.distort_image_and_masks_with_autoaugment(
          image, boxes, masks, self._autoaugment_policy_name)

    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = input_utils.random_horizontal_flip(
            image, boxes, masks)
      else:
        image, boxes = input_utils.random_horizontal_flip(
            image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = box_utils.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    # Now the coordinates of boxes are w.r.t the scaled image.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = input_utils.resize_and_crop_boxes(
        boxes, image_scale, image_info[1, :], offset)

    # Filters out ground truth boxes that are all zeros.
    indices = box_utils.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    if self._include_mask:
      masks = tf.gather(masks, indices)
      # Transfer boxes to the original image space and do normalization.
      cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
      cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
      cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_shape)
      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
          tf.expand_dims(masks, axis=-1),
          cropped_boxes,
          box_indices=tf.range(num_masks, dtype=tf.int32),
          crop_size=[self._mask_crop_size, self._mask_crop_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)

    if self._num_attributes:
      attributes = tf.gather(attributes, indices)

    # Assigns anchor targets.
    # Note that after the target assignment, box targets are absolute pixel
    # offsets w.r.t. the scaled image.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))
    anchor_labeler = anchor.RpnAnchorLabeler(
        input_anchor,
        self._rpn_match_threshold,
        self._rpn_unmatched_threshold,
        self._rpn_batch_size_per_im,
        self._rpn_fg_fraction)
    rpn_score_targets, rpn_box_targets = anchor_labeler.label_anchors(
        boxes, tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
        'rpn_score_targets': rpn_score_targets,
        'rpn_box_targets': rpn_box_targets,
    }
    labels['gt_boxes'] = input_utils.pad_to_fixed_size(
        boxes, self._max_num_instances, -1)
    labels['gt_classes'] = input_utils.pad_to_fixed_size(
        classes, self._max_num_instances, -1)

    if self._include_mask:
      labels['gt_masks'] = input_utils.pad_to_fixed_size(
          masks, self._max_num_instances, -1)

    if self._num_attributes:
      labels['gt_attributes'] = input_utils.pad_to_fixed_size(
          attributes, self._max_num_instances, -1)

    return image, labels

  def _parse_train_data_v2(self, data):
    """Parses data for training.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        rpn_score_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location]. The height_l and
          width_l represent the dimension of class logits at l-th level.
        rpn_box_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        gt_boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        gt_classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        gt_masks: groundtrugh masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
    """
    if self._use_autoaugment:
      try:
        from utils import autoaugment_utils  # pylint: disable=g-import-not-at-top
      except ImportError as e:
        logging.exception('Autoaugment is not supported in TF 2.x.')
        raise e

    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    masks = None
    attributes = None

    if self._include_mask:
      masks = data['groundtruth_instance_masks']

    if self._num_attributes:
      attributes = data['groundtruth_attributes']

    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training and self._is_training:
      num_groundtrtuhs = tf.shape(classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
            tf.greater(tf.size(is_crowds), 0),
            lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)

      if self._include_mask:
        masks = tf.gather(masks, indices)

      if self._num_attributes:
        attributes = tf.gather(attributes, indices)

    # Gets original image and its size.
    image = data['image']

    # NOTE: The autoaugment method works best when used alongside the standard
    # horizontal flipping of images along with size jittering and normalization.
    if self._use_autoaugment and not self._apply_autoaugment_after_resizing:
      image, boxes, masks = autoaugment_utils.distort_image_and_masks_with_autoaugment(
          image, boxes, masks, self._autoaugment_policy_name)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      if self._include_mask:
        image, boxes, masks = input_utils.random_horizontal_flip(
            image, boxes, masks)
      else:
        image, boxes = input_utils.random_horizontal_flip(
            image, boxes)

    # Resizes and crops image.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image, image_info = input_utils.resize_and_crop_image(image,
                                                          self._output_size,
                                                          aug_scale_min=self._aug_scale_min,
                                                          aug_scale_max=self._aug_scale_max)

    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    orig_image_shape = image_info[0]
    boxes = box_utils.denormalize_boxes(boxes, orig_image_shape)

    # Resizes and crops boxes.
    # Now the coordinates of boxes are w.r.t the scaled image.
    rescaled_image_shape = tf.shape(image)[:2]
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, rescaled_image_shape, offset)

    # Filters out ground truth boxes that are all zeros.
    boxes, classes, masks, attributes = self._remove_empty_boxes(boxes, classes, masks, attributes)

    # apply the autoaugment after resizing
    if self._use_autoaugment and self._apply_autoaugment_after_resizing:
      # prepare image and boxes for the autoaugment
      image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
      boxes = box_utils.normalize_boxes(boxes, rescaled_image_shape)

      # prepare masks for the autoaugment
      masks = tf.expand_dims(masks, axis=-1)
      scaled_mask_size = tf.cast(tf.round(orig_image_shape * image_scale), tf.int32)
      scaled_masks = tf.image.resize_images(masks, scaled_mask_size, method=tf.image.ResizeMethod.BILINEAR)
      offset_int = tf.cast(offset, tf.int32)
      masks = scaled_masks[:, offset_int[0]:offset_int[0] + rescaled_image_shape[0],
                           offset_int[1]:offset_int[1] + rescaled_image_shape[1]]
      masks = tf.squeeze(masks, axis=-1)
      masks = tf.cast(tf.greater(masks, 0.5), tf.float32)

      # apply the autoaugment
      image, boxes, masks = autoaugment_utils.distort_image_and_masks_with_autoaugment(
          image, boxes, masks, self._autoaugment_policy_name)

      # convert the image back to float32 and denormalize bboxes
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      boxes = box_utils.denormalize_boxes(boxes, rescaled_image_shape)

      # filters out empty bboxes
      boxes, classes, masks, attributes = self._remove_empty_boxes(boxes, classes, masks, attributes)

    if self._include_mask:
      if self._use_autoaugment and self._apply_autoaugment_after_resizing:
        # don't rescale boxes as masks were already resized
        cropped_boxes = box_utils.normalize_boxes(boxes, rescaled_image_shape)
      else:
        # transfer boxes to the original image space
        cropped_boxes = boxes + tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
        cropped_boxes /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        cropped_boxes = box_utils.normalize_boxes(cropped_boxes, orig_image_shape)

      num_masks = tf.shape(masks)[0]
      masks = tf.image.crop_and_resize(
          tf.expand_dims(masks, axis=-1),
          cropped_boxes,
          box_indices=tf.range(num_masks, dtype=tf.int32),
          crop_size=[self._mask_crop_size, self._mask_crop_size],
          method='bilinear')
      masks = tf.squeeze(masks, axis=-1)

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # pad the image
    padded_size = input_utils.compute_padded_size(self._output_size, 2 ** self._max_level)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_size[0], padded_size[1])

    image_height, image_width, _ = image.get_shape().as_list()

    # Assigns anchor targets.
    # Note that after the target assignment, box targets are absolute pixel
    # offsets w.r.t. the scaled image.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))
    anchor_labeler = anchor.RpnAnchorLabeler(
        input_anchor,
        self._rpn_match_threshold,
        self._rpn_unmatched_threshold,
        self._rpn_batch_size_per_im,
        self._rpn_fg_fraction)
    rpn_score_targets, rpn_box_targets = anchor_labeler.label_anchors(
        boxes, tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
        'rpn_score_targets': rpn_score_targets,
        'rpn_box_targets': rpn_box_targets,
    }
    labels['gt_boxes'] = input_utils.pad_to_fixed_size(
        boxes, self._max_num_instances, -1)
    labels['gt_classes'] = input_utils.pad_to_fixed_size(
        classes, self._max_num_instances, -1)

    if self._include_mask:
      labels['gt_masks'] = input_utils.pad_to_fixed_size(
          masks, self._max_num_instances, -1)

    if self._num_attributes:
      labels['gt_attributes'] = input_utils.pad_to_fixed_size(
          attributes, self._max_num_instances, -1)

    return image, labels

  def _remove_empty_boxes(self, boxes, classes, masks=None, attributes=None):
    """Filters out ground truth boxes that are all zeros."""
    indices = box_utils.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    if masks is not None:
        masks = tf.gather(masks, indices)

    if attributes is not None:
        attributes = tf.gather(attributes, indices)

    return boxes, classes, masks, attributes

  def _parse_eval_data(self, data):
    """Parses data for evaluation.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        groundtruths:
          source_id: Groundtruth source id.
          height: Original image height.
          width: Original image width.
          boxes: Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
             image that is fed to the network. The tennsor is padded with -1 to
             the fixed dimension [self._max_num_instances, 4].
          classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          areas: Box area or mask area depend on whether mask is present.
          is_crowds: Whether the ground truth label is a crowd label.
          num_groundtruths: Number of ground truths in the image.
    """
    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    # Assigns anchor targets.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Sets up groundtruth data for evaluation.
    groundtruths = {
        'source_id':
            data['source_id'],
        'height':
            data['height'],
        'width':
            data['width'],
        'num_groundtruths':
            tf.shape(data['groundtruth_classes']),
        'boxes':
            box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape),
        'classes':
            data['groundtruth_classes'],
        'areas':
            data['groundtruth_area'],
        'is_crowds':
            tf.cast(data['groundtruth_is_crowd'], tf.int32),
    }
    # TODO(b/143766089): Add ground truth masks for segmentation metrics.
    groundtruths['source_id'] = dataloader_utils.process_source_id(
        groundtruths['source_id'])
    groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(
        groundtruths, self._max_num_instances)

    # Packs labels for model_fn outputs.
    labels = {
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
        'groundtruths': groundtruths,
    }

    return image, labels

  def _parse_predict_data(self, data):
    """Parses data for prediction.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      A dictionary of {'images': image, 'labels': labels} where
        images: image tensor that is preproessed to have normalized value and
          dimension [output_size[0], output_size[1], 3]
        labels: a dictionary of tensors used for training. The following
          describes {key: value} pairs in the dictionary.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
          image_info: a 2D `Tensor` that encodes the information of the image
            and the applied preprocessing. It is in the format of
            [[original_height, original_width], [scaled_height, scaled_width],
          anchor_boxes: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, 4] representing anchor boxes at each
            level.
    """
    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = input_utils.normalize_image(image)

    # Resizes and crops image.
    image, image_info = input_utils.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=input_utils.compute_padded_size(
            self._output_size, 2 ** self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_height, image_width, _ = image.get_shape().as_list()

    # If bfloat16 is used, casts input image to tf.bfloat16.
    if self._use_bfloat16:
      image = tf.cast(image, dtype=tf.bfloat16)

    # Compute Anchor boxes.
    input_anchor = anchor.Anchor(
        self._min_level,
        self._max_level,
        self._num_scales,
        self._aspect_ratios,
        self._anchor_size,
        (image_height, image_width))

    labels = {
        'source_id': dataloader_utils.process_source_id(data['source_id']),
        'anchor_boxes': input_anchor.multilevel_boxes,
        'image_info': image_info,
    }

    if self._mode == ModeKeys.PREDICT_WITH_GT:
      # Converts boxes from normalized coordinates to pixel coordinates.
      boxes = box_utils.denormalize_boxes(
          data['groundtruth_boxes'], image_shape)
      groundtruths = {
          'source_id': data['source_id'],
          'height': data['height'],
          'width': data['width'],
          'num_detections': tf.shape(data['groundtruth_classes']),
          'boxes': boxes,
          'classes': data['groundtruth_classes'],
          'areas': data['groundtruth_area'],
          'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
      }
      groundtruths['source_id'] = dataloader_utils.process_source_id(
          groundtruths['source_id'])
      groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(
          groundtruths, self._max_num_instances)
      labels['groundtruths'] = groundtruths

    return {
        'images': image,
        'labels': labels,
    }
