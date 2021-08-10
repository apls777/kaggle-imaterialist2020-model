# from __future__ import annotations

from dataclasses import dataclass

# from pathlib import Path
from typing import NewType

# import numpy as np
# from six import with_metaclass
# from utils import input_utils

# import tensorflow_core._api.v1.compat.v1 as tf
# import yaml
# from configs import factory as config_factory
# from hyperparameters import params_dict
# from modeling import factory as model_factory
# from PIL import Image
from typing_extensions import TypedDict
from evaluation import coco_utils

Height = NewType("Height", int)
Width = NewType("Width", int)
RLE = NewType("RLE", str)


class COCORLE(TypedDict):
    size: tuple[Height, Width]
    counts: RLE


@dataclass(frozen=True)
class Segmentation:
    filename: str
    segmentation: COCORLE
    imat_category_id: int
    score: float
    mask_mean_score: float


# # def image_download_loop(client: storage.Client, inputs: list, out_queue):
# #     image = client.download()
# #     out_queue.put(image)


# def build_params(model_dir: Path, params_override: str) -> params_dict.ParamsDict:
#     config_file = model_dir / "params.yaml"
#     with config_file.open() as f:
#         model_type = yaml.load(f, Loader=yaml.CLoader)["type"]

#     params = config_factory.config_generator(model_type)
#     if config_file.is_file():
#         params = params_dict.override_params_dict(
#             params, str(config_file), is_strict=True
#         )
#         # parmas_override is a YAML/JSON string or a YAML file path (str)
#         # https://github.com/tensorflow/models/blob/master/official/common/flags.py#L72-L84
#         params = params_dict.override_params_dict(
#             params, params_override, is_strict=True
#         )
#     params.validate()
#     params.lock()
#     return params


# def segmentation_loop(
#     model_dir: Path, params_override: str | None, in_queue, batch_size: int, out_queue, image_size: int = 640,
# ):
#     # load model
#     params = build_params(
#         model_dir=model_dir,
#         params_override=params_override,
#     )
#     model = model_factory.model_generator(params)

#     with tf.Graph().as_default():
#         image_input = tf.placeholder(shape=(), dtype=tf.string)
#         image = tf.io.decode_image(image_input, channels=3)
#         image.set_shape([None, None, 3])

#         image = input_utils.normalize_image(image)
#         _image_size = [image_size, image_size]
#         image, image_info = input_utils.resize_and_crop_image(
#             image=image,
#             desired_size=_image_size,
#             padded_size=_image_size,
#             aug_scale_min=1.0,
#             aug_scale_max=1.0)
#         # image: (height, width, RGB)
#         # image_info: (4, 2)=(original|desired|scale|offset, height(y)|width(x))
#         image.set_shape([image_size[0], image_size[1], 3])

#         # batching.
#         images = tf.reshape(image, [batch_size, image_size[0], image_size[1], 3])
#         images_info = tf.expand_dims(image_info, axis=0)
#         # (1, 4, 2)

#         # model inference
#         outputs = model.build_outputs(
#             images, {'image_info': images_info}, mode=mode_keys.PREDICT)

#         outputs['detection_boxes'] = (
#             outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

#         predictions = outputs

#         # Create a saver in order to load the pre-trained checkpoint.
#         saver = tf.train.Saver()

#         image_with_detections_list = []
#         with tf.Session() as sess:
#             print(' - Loading the checkpoint...')
#             saver.restore(sess, FLAGS.checkpoint_path)

#             image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
#             for i, image_file in enumerate(image_files):

#     stop = True
#     while stop:
#         image = in_queue.get()
#         segmentations = _segment(image)
#         out_queue.put(segmentations)


# # def bq_insertion_loop(client = bigquery.Client, in_queue):
# #     stop = True
# #     while stop:
# #         segmentations = in_queue.get()
# #     client.insert_rows(segmentations)


# def _segment(sess: tf.Session, images: list[Image.Image]) -> list[Segmentation]:
#     for image in images:
#         width, height = image.size
#         np_image = np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8)

#         predictions_np = sess.run(predictions, feed_dict={image_input: image_bytes})

#         num_detections = int(predictions_np["num_detections"][0])
#         np_boxes = predictions_np["detection_boxes"][0, :num_detections]
#         np_scores = predictions_np["detection_scores"][0, :num_detections]
#         np_classes = predictions_np["detection_classes"][0, :num_detections]
#         np_classes = np_classes.astype(np.int32)
#         np_masks = None
#         if "detection_masks" in predictions_np:
#             instance_masks = predictions_np["detection_masks"][0, :num_detections]
#             np_masks = mask_utils.paste_instance_masks(
#                 instance_masks, box_utils.yxyx_to_xywh(np_boxes), height, width
#             )

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow_core._api.v1.compat.v1 as tf

from configs import factory as config_factory
from dataloader import mode_keys
from modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model", "retinanet", "Support `retinanet`, `mask_rcnn` and `shapemask`."
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
    "label_map_file",
    "",
    "The label map file. See --label_map_format for the definition.",
)
flags.DEFINE_string(
    "label_map_format",
    "csv",
    "The format of the label map file. Currently only support `csv` where the "
    "format of each row is: `id:name`.",
)
flags.DEFINE_string(
    "image_file_pattern", "", "The glob that specifies the image file pattern."
)
flags.DEFINE_string(
    "out_json",
    "/tmp/segment_from_gcs_to_bq.json",
    "The output HTML file that includes images with rendered detections.",
)
flags.DEFINE_integer("max_boxes_to_draw", 10, "The maximum number of boxes to draw.")
flags.DEFINE_float(
    "min_score_threshold", 0.05, "The minimum score thresholds in order to draw boxes."
)


def main(unused_argv):
    del unused_argv
    # Load the label map.
    print(" - Loading the label map...")
    label_map_dict = {}
    if FLAGS.label_map_format == "csv":
        with tf.gfile.Open(FLAGS.label_map_file, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=":")
            for row in reader:
                if len(row) != 2:
                    raise ValueError(
                        "Each row of the csv label map file must be in "
                        "`id:name` format."
                    )
                id_index = int(row[0])
                name = row[1]
                label_map_dict[id_index] = {
                    "id": id_index,
                    "name": name,
                }
    else:
        raise ValueError(
            "Unsupported label map format: {}.".format(FLAGS.label_mape_format)
        )

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
        images_info = tf.expand_dims(image_info, axis=0)

        # model inference
        outputs = model.build_outputs(
            images, {"image_info": images_info}, mode=mode_keys.PREDICT
        )

        outputs["detection_boxes"] = outputs["detection_boxes"] / tf.tile(
            images_info[:, 2:3, :], [1, 1, 2]
        )

        predictions = outputs

        # Create a saver in order to load the pre-trained checkpoint.
        saver = tf.train.Saver()

        image_with_detections_list = []
        with tf.Session() as sess:
            print(" - Loading the checkpoint...")
            saver.restore(sess, FLAGS.checkpoint_path)

            image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
            for i, image_file in enumerate(image_files):
                print(" - Processing image %d..." % i)

                with tf.gfile.GFile(image_file, "rb") as f:
                    image_bytes = f.read()

                image = Image.open(image_file)
                image = image.convert("RGB")  # needed for images with 4 channels.
                width, height = image.size
                np_image = (
                    np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8)
                )

                predictions_np = sess.run(
                    predictions, feed_dict={image_input: image_bytes}
                )

                predictions = coco_utils.convert_predictions_to_coco_annotations(
                    predictions_np,
                    output_image_size=1024,
                    encode_mask_fn=encode_mask_fn,
                    score_threshold=self._score_threshold,
                )

                seg = Segmentation(
                    filename="foo", # predictions["image_id"] から変換できそう
                    segmentation=predictions["segmentation"],
                    imat_category_id=predictions["category_id"],
                    score=predictions["score"],
                    mask_mean_score=predictions["mask_mean_score"],
                )

    print(" - Saving the outputs...")
    json_str = ""
    with tf.gfile.GFile(FLAGS.out_json, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("label_map_file")
    flags.mark_flag_as_required("image_file_pattern")
    flags.mark_flag_as_required("output_html")
    logging.set_verbosity(logging.INFO)
    tf.app.run(main)
