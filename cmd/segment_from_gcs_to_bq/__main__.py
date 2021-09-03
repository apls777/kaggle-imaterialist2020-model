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
flags.DEFINE_integer("batch_size", 2, "The batch size.")
flags.DEFINE_string("checkpoint_path", "", "The path to the checkpoint file.")
flags.DEFINE_string("config_file", "", "The config file template.")
flags.DEFINE_string(
    "params_override",
    "",
    "The YAML file/string that specifies the parameters "
    "override in addition to the `config_file`.",
)
flags.DEFINE_string(
    "image_file_pattern", "", "The glob that specifies the image file pattern."
)
flags.DEFINE_float(
    "min_score_threshold", 0.05, "The minimum score thresholds in order to draw boxes."
)


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [FLAGS.image_size, FLAGS.image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image, image_size, image_size, aug_scale_min=1.0, aug_scale_max=1.0
    )
    image.set_shape([image_size[0], image_size[1], 3])

    return image, image_info


def input_fn(data_files_pattern, batch_size):
    filenames = tf.io.gfile.glob(data_files_pattern)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


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
        images = tf.placeholder(
            shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3),
            dtype=tf.float32,
        )
        # images: (batch_size, width, height, RGB=3)

        image_infos = tf.placeholder(shape=(FLAGS.batch_size, 4, 2), dtype=tf.float32)
        # image_infos: (batch_size, original|desired|scale|offset=4, height(y)|width(x)=2)  # noqa: E501

        # model inference
        outputs = model.build_outputs(
            images,
            {"image_info": image_infos},
            mode=mode_keys.PREDICT,
        )

        outputs["detection_boxes"] = outputs["detection_boxes"] / tf.tile(
            image_infos[:, 2:3, :], [1, 1, 2]
        )

        predictions = model.build_predictions(outputs, {"image_info": image_infos})

        source_id = tf.placeholder(dtype=tf.int32)
        predictions["pred_source_id"] = tf.expand_dims(source_id, axis=0)

        # Create a saver in order to load the pre-trained checkpoint.
        saver = tf.train.Saver()

        image_files_pattern = f"{FLAGS.image_file_pattern.rstrip('/')}/*"
        print(image_files_pattern)
        # label_to_index = dict(
        #    (index, os.path.basename(name)) for index, name in enumerate(image_files)
        # )
        # print(label_to_index)
        # for source_index, image_file in enumerate(image_files):

        print(" - Processing image ...")

        dataset = input_fn(
            data_files_pattern=image_files_pattern, batch_size=FLAGS.batch_size
        )
        print(dataset)
        itr = tf.data.make_one_shot_iterator(dataset)

        print("**************")

        with tf.Session() as sess:
            print(" - Loading the checkpoint...")
            saver.restore(sess, FLAGS.checkpoint_path)

            while True:
                try:
                    image_batch, image_info_batch = itr.get_next()
                    print("image_batch: ", image_batch)
                    print("image_info_batch: ", image_info_batch)

                    image_batch_np = sess.run(image_batch)
                    image_info_batch_np = sess.run(image_info_batch)
                    predictions_np = sess.run(
                        predictions,
                        feed_dict={
                            images: image_batch_np,
                            image_infos: image_info_batch_np,
                            source_id: [1, 2],
                        },
                    )
                    print(" --- INFERENCE FINISHED")
                    print(predictions_np)

                    predictions_ = {}
                    for key, val in predictions_np.items():
                        if key[0:5] == "pred_":
                            predictions_[key[5::]] = val

                    for k, v in six.iteritems(predictions_):
                        predictions_[k] = np.expand_dims(predictions_[k], axis=0)

                    coco_annotations = (
                        coco_utils.convert_predictions_to_coco_annotations(
                            predictions_,
                            output_image_size=1024,
                            score_threshold=FLAGS.min_score_threshold,
                        )
                    )

                    seg_list = []

                    for j in range(len(coco_annotations)):

                        # In the case of byte type, it cannot be converted to json
                        coco_annotations[j]["segmentation"]["counts"] = str(
                            coco_annotations[j]["segmentation"]["counts"]
                        )

                        seg = Segmentation(
                            image_id=coco_annotations[j]["image_id"],
                            filename=os.path.basename(image_files_pattern),
                            segmentation=coco_annotations[j]["segmentation"],
                            imat_category_id=coco_annotations[j]["category_id"],
                            score=coco_annotations[j]["score"],
                            mask_mean_score=coco_annotations[j]["mask_mean_score"],
                        )

                    seg_list.append(asdict(seg))

                    print(seg_list)
                except tf.errors.OutOfRangeError as e:
                    print(e)
                    break


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("image_file_pattern")
    logging.set_verbosity(logging.INFO)
    tf.app.run(main)
