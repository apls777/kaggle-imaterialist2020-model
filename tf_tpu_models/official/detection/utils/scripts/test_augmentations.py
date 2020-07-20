# Lint as: python2, python3
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
"""Main function to train various models.
"""

from absl import flags
from absl import logging

from six.moves import range
import tensorflow.compat.v1 as tf

from configs import factory
from dataloader import input_reader
from dataloader import mode_keys as ModeKeys

from hyperparameters import common_hparams_flags
from hyperparameters import params_dict
from utils import mask_utils, box_utils
from utils.input_utils import denormalize_image
from utils.object_detection.visualization_utils import draw_bounding_boxes_on_image_tensors
from utils.paths import root_dir

common_hparams_flags.define_common_hparams_flags()

flags.DEFINE_string(
    'model', default='retinanet',
    help='Support `retinanet`, `mask_rcnn`, `shapemask` and `classification`.')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    params = factory.config_generator(FLAGS.model)

    if FLAGS.config_file:
        params = params_dict.override_params_dict(
            params, FLAGS.config_file, is_strict=True)

    params = params_dict.override_params_dict(
        params, FLAGS.params_override, is_strict=True)

    params.train.input_partition_dims = None
    params.train.num_cores_per_replica = None
    params.architecture.use_bfloat16 = False
    # params.maskrcnn_parser.use_autoaugment = False

    params.validate()
    params.lock()

    # Prepares input functions for train and eval.
    train_input_fn = input_reader.InputFnTest(
        params.train.train_file_pattern, params, mode=ModeKeys.TRAIN,
        dataset_type=params.train.train_dataset_type)

    batch_size = 1
    dataset = train_input_fn({'batch_size': batch_size})

    category_index = {}
    for i in range(50):
        category_index[i] = {
            'name': 'test_%d' % i,
        }

    for i, (image_batch, labels_batch) in enumerate(dataset.take(10)):
        image_batch = tf.transpose(image_batch, [3, 0, 1, 2])
        image_batch = tf.map_fn(denormalize_image, image_batch, dtype=tf.uint8, back_prop=False)

        image_shape = tf.shape(image_batch)[1:3]

        masks_batch = []
        for image, bboxes, masks in zip(image_batch, labels_batch['gt_boxes'], labels_batch['gt_masks']):
            # extract masks
            bboxes = tf.numpy_function(box_utils.yxyx_to_xywh, [bboxes], tf.float32)
            binary_masks = tf.numpy_function(mask_utils.paste_instance_masks,
                                             [masks, bboxes, image_shape[0], image_shape[1]],
                                             tf.uint8)

            masks_batch.append(binary_masks)

        masks_batch = tf.stack(masks_batch, axis=0)

        scores_mask = tf.cast(tf.greater(labels_batch['gt_classes'], -1), tf.float32)
        scores = tf.ones_like(labels_batch['gt_classes'], dtype=tf.float32) * scores_mask

        images = draw_bounding_boxes_on_image_tensors(image_batch,
                                                      labels_batch['gt_boxes'],
                                                      labels_batch['gt_classes'],
                                                      scores,
                                                      category_index,
                                                      instance_masks=masks_batch,
                                                      use_normalized_coordinates=False)

        for j, image in enumerate(images):
            image_bytes = tf.io.encode_jpeg(image)
            tf.io.write_file(root_dir('data/visualizations/aug_%d.jpg' % (i * batch_size + j)), image_bytes)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    tf.enable_eager_execution()
    tf.app.run(main)
