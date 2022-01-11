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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf


def _get_source_id_from_encoded_image(parsed_tensors):
    return tf.strings.as_string(
        tf.strings.to_hash_bucket_fast(parsed_tensors["image/encoded"], 2 ** 63 - 1)
    )


class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(
        self, include_mask=False, regenerate_source_id=False, num_attributes=None
    ):
        self._include_mask = include_mask
        self._regenerate_source_id = regenerate_source_id
        self._num_attributes = num_attributes
        self._keys_to_features = {
            "image/encoded": tf.io.FixedLenFeature((), tf.string),
            "image/source_id": tf.io.FixedLenFeature((), tf.string, ""),
            "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
            "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/area": tf.io.VarLenFeature(tf.float32),
            "image/object/is_crowd": tf.io.VarLenFeature(tf.int64),
        }

        if include_mask:
            self._keys_to_features.update(
                {
                    "image/object/mask": tf.io.VarLenFeature(tf.string),
                }
            )

        if num_attributes:
            self._keys_to_features.update(
                {
                    "image/object/attributes/labels": tf.io.FixedLenFeature(
                        (), tf.string, ""
                    ),
                }
            )

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(parsed_tensors["image/encoded"], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_masks(self, parsed_tensors):
        """Decode a set of PNG masks to the tf.float32 tensors."""

        def _decode_png_mask(png_bytes):
            mask = tf.squeeze(
                tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1
            )
            mask = tf.cast(mask, dtype=tf.float32)
            mask.set_shape([None, None])
            return mask

        height = parsed_tensors["image/height"]
        width = parsed_tensors["image/width"]
        masks = parsed_tensors["image/object/mask"]
        return tf.cond(
            pred=tf.greater(tf.size(input=masks), 0),
            true_fn=lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
            false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32),
        )

    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.cond(
            pred=tf.greater(tf.shape(input=parsed_tensors["image/object/area"])[0], 0),
            true_fn=lambda: parsed_tensors["image/object/area"],
            false_fn=lambda: (xmax - xmin) * (ymax - ymin),
        )

    def decode(self, serialized_example, max_num_instances: int = None):
        """Decode the serialized example.

        Args:
          serialized_example: a single serialized tf.Example string.

        Returns:
          decoded_tensors: a dictionary of tensors with the following fields:
            - image: a uint8 tensor of shape [None, None, 3].
            - source_id: a string scalar tensor.
            - height: an integer scalar tensor.
            - width: an integer scalar tensor.
            - groundtruth_classes: a int64 tensor of shape [None].
            - groundtruth_is_crowd: a bool tensor of shape [None].
            - groundtruth_area: a float32 tensor of shape [None].
            - groundtruth_boxes: a float32 tensor of shape [None, 4].
            - groundtruth_instance_masks: a float32 tensor of shape
                [None, None, None].
            - groundtruth_instance_masks_png: a string tensor of shape [None].
            - groundtruth_attributes - an int32 tensor of shape [None, num_attributes]
        """
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features
        )
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0
                    )

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        areas = self._decode_areas(parsed_tensors)

        decode_image_shape = tf.logical_or(
            tf.equal(parsed_tensors["image/height"], -1),
            tf.equal(parsed_tensors["image/width"], -1),
        )
        image_shape = tf.cast(tf.shape(input=image), dtype=tf.int64)

        parsed_tensors["image/height"] = tf.compat.v1.where(
            decode_image_shape, image_shape[0], parsed_tensors["image/height"]
        )
        parsed_tensors["image/width"] = tf.compat.v1.where(
            decode_image_shape, image_shape[1], parsed_tensors["image/width"]
        )

        is_crowds = tf.cond(
            pred=tf.greater(
                tf.shape(input=parsed_tensors["image/object/is_crowd"])[0], 0
            ),
            true_fn=lambda: tf.cast(
                parsed_tensors["image/object/is_crowd"], dtype=tf.bool
            ),
            false_fn=lambda: tf.zeros_like(
                parsed_tensors["image/object/class/label"], dtype=tf.bool
            ),
        )  # pylint: disable=line-too-long
        if self._regenerate_source_id:
            source_id = _get_source_id_from_encoded_image(parsed_tensors)
        else:
            source_id = tf.cond(
                pred=tf.greater(
                    tf.strings.length(input=parsed_tensors["image/source_id"]), 0
                ),
                true_fn=lambda: parsed_tensors["image/source_id"],
                false_fn=lambda: _get_source_id_from_encoded_image(parsed_tensors),
            )
        if self._include_mask:
            masks = self._decode_masks(parsed_tensors)

        decoded_tensors = {
            "image": image,
            "source_id": source_id,
            "height": parsed_tensors["image/height"],
            "width": parsed_tensors["image/width"],
            "groundtruth_classes": parsed_tensors["image/object/class/label"],
            "groundtruth_is_crowd": is_crowds,
            "groundtruth_area": areas,
            "groundtruth_boxes": boxes,
        }

        if self._include_mask:
            decoded_tensors.update(
                {
                    "groundtruth_instance_masks": masks,
                    "groundtruth_instance_masks_png": parsed_tensors[
                        "image/object/mask"
                    ],
                }
            )

        if self._num_attributes:
            decoded_tensors.update(
                {
                    "groundtruth_attributes": tf.reshape(
                        tf.cast(
                            tf.io.decode_raw(
                                parsed_tensors["image/object/attributes/labels"],
                                tf.bool,
                            ),
                            tf.float32,
                        ),
                        shape=(-1, self._num_attributes),
                    ),
                }
            )

        if max_num_instances:
            for key, value in decoded_tensors.items():
                if key.startswith("groundtruth_"):
                    decoded_tensors[key] = value[:max_num_instances]

        return decoded_tensors
