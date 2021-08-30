# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=line-too-long
r"""A stand-alone binary to run model inference and visualize results.

It currently only supports model of type `retinanet` and `mask_rcnn`. It only
supports running on CPU/GPU with batch size 1.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import, annotations, division, print_function

import os
from dataclasses import asdict, dataclass
from typing import NewType

import numpy as np
import six
import tensorflow_core._api.v1.compat.v1 as tf
from absl import flags, logging
from configs import factory as config_factory
from dataloader import mode_keys
from evaluation import coco_utils
from hyperparameters import params_dict
from modeling import factory as model_factory
from typing_extensions import TypedDict
from utils import input_utils

Height = NewType("Height", int)
Width = NewType("Width", int)
RLE = NewType("RLE", str)


class COCORLE(TypedDict):
    size: tuple[Height, Width]
    counts: RLE


@dataclass(frozen=True)
class Segmentation:
    image_id: int
    filename: str
    segmentation: COCORLE
    imat_category_id: int
    score: float
    mask_mean_score: float


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model", "mask_rcnn", "Support `retinanet`, `mask_rcnn` and `shapemask`."
)
flags.DEFINE_integer("image_size", 640, "The image size.")
flags.DEFINE_string("checkpoint_path", "", "The path to the checkpoint file.")
flags.DEFINE_string("config_file", "", "The config file template.")
flags.DEFINE_string(
    "params_override",
    "",
    "The YAML file/string that specifies the parameters "
    "override in addition to the `config_file`.",
)
flags.DEFINE_string(
    "image_files_pattern", "", "The glob that specifies the image file pattern."
)
flags.DEFINE_float(
    "min_score_threshold", 0.05, "The minimum score thresholds in order to draw boxes."
)


def main(unused_argv):
    del unused_argv
    # Load the label map.
    print(" - Loading the label map...")

    params = config_factory.config_generator(FLAGS.model)
    if FLAGS.config_file:
        params = params_dict.override_params_dict(
            params, FLAGS.config_file, is_strict=True
        )
    params = params_dict.override_params_dict(
        params, FLAGS.params_override, is_strict=True
    )
    params.validate()
    params.lock()

    model = model_factory.model_generator(params)

    with tf.Graph().as_default():
        image_input = tf.placeholder(shape=(), dtype=tf.string)
        image = tf.io.decode_image(image_input, channels=3)
        image.set_shape([None, None, 3])

        image = input_utils.normalize_image(image)
        image_size = [FLAGS.image_size, FLAGS.image_size]
        image, image_info = input_utils.resize_and_crop_image(
            image, image_size, image_size, aug_scale_min=1.0, aug_scale_max=1.0
        )
        image.set_shape([image_size[0], image_size[1], 3])

        # batching.
        images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
        # images: (batch_size=2, width, height, RGB=3)
        images_info = tf.expand_dims(image_info, axis=0)
        # image_info: (2, 4, 2)=(batch_size, original|desired|scale|offset, height(y)|width(x))  # noqa: E501

        # model inference
        outputs = model.build_outputs(
            images,
            {"image_info": images_info},
            mode=mode_keys.PREDICT,
        )

        outputs["detection_boxes"] = outputs["detection_boxes"] / tf.tile(
            images_info[:, 2:3, :], [1, 1, 2]
        )

        # outputs["image_info"] = images_info
        # predictions = outputs
        predictions = model.build_predictions(outputs, {"image_info": images_info})
        source_id = tf.placeholder(dtype=tf.int32)
        predictions["pred_source_id"] = tf.expand_dims(source_id, axis=0)

        # Create a saver in order to load the pre-trained checkpoint.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print(" - Loading the checkpoint...")
            saver.restore(sess, FLAGS.checkpoint_path)

            image_files = tf.gfile.Glob(f"{FLAGS.image_files_pattern.rstrip('/')}/*")
            for source_index, image_file in enumerate(image_files):

                print(f" - Processing image {source_index}...")
                print(os.path.basename(image_file))

                with tf.gfile.GFile(image_file, "rb") as f:
                    image_bytes = f.read()

                predictions_np = sess.run(
                    predictions,
                    feed_dict={image_input: image_bytes, source_id: source_index},
                )

                predictions_ = {}
                for key, val in predictions_np.items():
                    # print(key, val)
                    if key[0:5] == "pred_":
                        predictions_[key[5::]] = val

                for k, v in six.iteritems(predictions_):
                    predictions_[k] = np.expand_dims(predictions_[k], axis=0)

                coco_annotations = coco_utils.convert_predictions_to_coco_annotations(
                    predictions_,
                    output_image_size=1024,
                    score_threshold=FLAGS.min_score_threshold,
                )

                seg_list = []

                for j in range(len(coco_annotations)):
                    seg = Segmentation(
                        image_id=coco_annotations[j]["image_id"],
                        filename=os.path.basename(image_file),
                        segmentation=coco_annotations[j]["segmentation"],
                        imat_category_id=coco_annotations[j]["category_id"],
                        score=coco_annotations[j]["score"],
                        mask_mean_score=coco_annotations[j]["mask_mean_score"],
                    )

                    seg_list.append(asdict(seg))

                print(seg_list)


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("image_files_pattern")
    logging.set_verbosity(logging.INFO)
    tf.app.run(main)
