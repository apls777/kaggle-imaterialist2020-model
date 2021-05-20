# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Convert raw COCO dataset to TFRecord for object_detection.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
from __future__ import absolute_import, division, print_function

import collections
import hashlib
import io
import json
import logging
import multiprocessing
import os
import tempfile
from typing import List
from pathlib import Path

import numpy as np
import PIL.Image
import tensorflow_core._api.v1.compat.v1 as tf
from absl import app, flags
from pycocotools import mask
from research.object_detection.utils import dataset_util, label_map_util

flags.DEFINE_boolean(
    "include_masks",
    False,
    "Whether to include instance segmentations masks "
    "(PNG encoded) in the result. default: False.",
)
flags.DEFINE_string("image_dir", "", "Directory containing images.")
flags.DEFINE_string(
    "image_info_file",
    "",
    "File containing image information. "
    "Tf Examples in the output files correspond to the image "
    "info entries in this file. If this file is not provided "
    "object_annotations_file is used if present. Otherwise, "
    "caption_annotations_file is used to get image info.",
)
flags.DEFINE_string(
    "object_annotations_file",
    "",
    "File containing object " "annotations - boxes and instance masks.",
)
flags.DEFINE_string(
    "caption_annotations_file", "", "File containing image " "captions."
)
flags.DEFINE_string("output_file_prefix", "/tmp/train", "Path to output file")
flags.DEFINE_integer("num_shards", 32, "Number of shards for output file.")

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


# Hashing can't work.
# c.f., https://github.com/tensorflow/tpu/issues/917
def hash_image_id(image_id: str) -> int:
    # Truncate sha1 (20 bytes) to Python int (0 ~ 2^(4*7)-1).
    return int(hashlib.sha1(str(image_id).encode("utf-8")).hexdigest()[:7], 16)


def create_tf_example(
    image,
    image_dir,
    bbox_annotations=None,
    category_index=None,
    caption_annotations=None,
    include_masks=False,
    num_attributes=None,
):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
      image_dir: directory containing the image files.
      bbox_annotations:
        list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
          u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
          coordinates in the official COCO dataset are given as [x, y, width,
          height] tuples using absolute coordinates where x, y represent the
          top-left (0-indexed) corner.  This function converts to the format
          expected by the Tensorflow Object Detection API (which is which is
          [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
          size).
      category_index: a dict containing COCO category information keyed by the
        'id' field of each category.  See the label_map_util.create_category_index
        function.
      caption_annotations:
        list of dict with keys: [u'id', u'image_id', u'str'].
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.

    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image["height"]
    image_width = image["width"]
    filename = image["file_name"]
    image_id = image["id"]

    full_path = os.path.join(image_dir, filename)
    image = PIL.Image.open(full_path)
    print(
        f"Resize image {filename}: ({image.width}, {image.height}) -> ({image_width}, {image_height})"
    )
    image = image.resize((image_width, image_height))
    with tempfile.NamedTemporaryFile("wb", suffix=".jpg") as f:
        image.save(f.name)
        with tf.gfile.GFile(f.name, "rb") as fid:
            encoded_jpg = fid.read()

    assert (
        image_width == image.width and image_height == image.height
    ), f"filename={filename}: label width={image_width}, height={image_height} but actual width={image.width}, height={image.height}"

    key = hashlib.sha256(encoded_jpg).hexdigest()
    # Hashing can't work.
    # c.f., https://github.com/tensorflow/tpu/issues/917
    # image_id = hash_image_id(image_id)
    feature_dict = {
        "image/height": dataset_util.int64_feature(image_height),
        "image/width": dataset_util.int64_feature(image_width),
        "image/filename": dataset_util.bytes_feature(filename.encode("utf8")),
        # source_id must be integer string
        # c.f., https://github.com/tensorflow/tpu/issues/516
        # c.f., process_source_id in tf_tpu_models/official/deteciton/utils/dataloader_utils.py
        "image/source_id": dataset_util.bytes_feature(str(image_id).encode("utf-8")),
        "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature("jpeg".encode("utf8")),
    }

    num_annotations_skipped = 0
    if bbox_annotations:
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        is_crowd = []
        category_names = []
        category_ids = []
        attributes_multi_hot = (
            np.zeros((len(bbox_annotations), num_attributes), dtype=np.bool)
            if num_attributes
            else None
        )
        area = []
        encoded_mask_png = []
        for i, object_annotations in enumerate(bbox_annotations):
            (x, y, width, height) = tuple(object_annotations["bbox"])
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            if x + width > image_width or y + height > image_height:
                num_annotations_skipped += 1
                continue
            xmin.append(float(x) / image_width)
            xmax.append(float(x + width) / image_width)
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)
            is_crowd.append(object_annotations["iscrowd"])
            category_id = int(object_annotations["category_id"])
            category_ids.append(category_id)
            category_names.append(category_index[category_id]["name"].encode("utf8"))
            area.append(object_annotations["area"])

            if include_masks:
                segmentation = object_annotations["segmentation"]
                if isinstance(segmentation, list):
                    if isinstance(segmentation[0], int):
                        binary_mask = _get_binary_mask(
                            segmentation, image_height, image_width
                        )
                    elif isinstance(segmentation[0], list):
                        run_len_encoding = mask.frPyObjects(
                            segmentation, image_height, image_width
                        )
                        binary_mask = mask.decode(run_len_encoding)
                        if not object_annotations["iscrowd"] and (
                            len(binary_mask.shape) > 2
                        ):
                            binary_mask = np.amax(binary_mask, axis=2)
                elif (
                    isinstance(segmentation, dict)
                    and "counts" in segmentation.keys()
                    and "size" in segmentation.keys()
                ):
                    binary_mask = mask.decode(segmentation)
                    if not object_annotations["iscrowd"] and (
                        len(binary_mask.shape) > 2
                    ):
                        binary_mask = np.amax(binary_mask, axis=2)
                else:
                    raise ValueError(f"not supported format annotation: {segmentation}")

                pil_image = PIL.Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format="PNG")
                encoded_mask_png.append(output_io.getvalue())

            if num_attributes:
                attributes_multi_hot[i, object_annotations["attribute_ids"]] = 1

        feature_dict.update(
            {
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmin),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmax),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymin),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymax),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    category_names
                ),
                "image/object/class/label": dataset_util.int64_list_feature(
                    category_ids
                ),
                "image/object/attributes/labels": dataset_util.bytes_feature(
                    attributes_multi_hot.tobytes()
                ),
                "image/object/is_crowd": dataset_util.int64_list_feature(is_crowd),
                "image/object/area": dataset_util.float_list_feature(area),
            }
        )
        if include_masks:
            feature_dict["image/object/mask"] = dataset_util.bytes_list_feature(
                encoded_mask_png
            )
    if caption_annotations:
        captions = []
        for caption_annotation in caption_annotations:
            captions.append(caption_annotation["caption"].encode("utf8"))
        feature_dict.update(
            {"image/caption": dataset_util.bytes_list_feature(captions)}
        )

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
    return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
    """Loads object annotation JSON file."""
    with tf.gfile.GFile(object_annotations_file, "r") as fid:
        obj_annotations = json.load(fid)

    try:
        num_attributes = obj_annotations["info"]["num_attributes"]
    except KeyError:
        print(
            "Get `num_attributes` by adding 1 to the muximum attribute id because COCO JSON doesn't have `info.num_attributes`."
        )
        num_attributes = obj_annotations["attributes"][-1]["id"] + 1

    images = obj_annotations["images"]
    category_index = label_map_util.create_category_index(obj_annotations["categories"])

    img_to_obj_annotation = collections.defaultdict(list)
    logging.info("Building bounding box index.")
    for annotation in obj_annotations["annotations"]:
        image_id = annotation["image_id"]
        img_to_obj_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image["id"]
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1

    logging.info("%d images are missing bboxes.", missing_annotation_count)

    return img_to_obj_annotation, category_index, num_attributes


def _load_caption_annotations(caption_annotations_file):
    """Loads caption annotation JSON file."""
    with tf.gfile.GFile(caption_annotations_file, "r") as fid:
        caption_annotations = json.load(fid)

    img_to_caption_annotation = collections.defaultdict(list)
    logging.info("Building caption index.")
    for annotation in caption_annotations["annotations"]:
        image_id = annotation["image_id"]
        img_to_caption_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    images = caption_annotations["images"]
    for image in images:
        image_id = image["id"]
        if image_id not in img_to_caption_annotation:
            missing_annotation_count += 1

    logging.info("%d images are missing captions.", missing_annotation_count)

    return img_to_caption_annotation


def _load_images_info(images_info_file):
    with tf.gfile.GFile(images_info_file, "r") as fid:
        info_dict = json.load(fid)
    return info_dict["images"]


def _load_images_info_from_dir(image_dir: str):
    image_paths = sorted(list(Path(image_dir).glob("*")))
    for i, path in enumerate(image_paths):
        image = PIL.Image.open(path)
        yield {
            "id": i,
            "height": image.height,
            "width": image.width,
            "file_name": path.name,
        }


def _create_tf_record_from_coco_annotations(
    images_info_file,
    image_dir,
    output_path,
    num_shards,
    object_annotations_file=None,
    caption_annotations_file=None,
    include_masks=False,
):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
      images_info_file: JSON file containing image info. The number of tf.Examples
        in the output tf Record files is exactly equal to the number of image info
        entries in this file. This can be any of train/val/test annotation json
        files Eg. 'image_info_test-dev2017.json',
        'instance_annotations_train2017.json',
        'caption_annotations_train2017.json', etc.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      num_shards: Number of output files to create.
      object_annotations_file: JSON file containing bounding box annotations.
      caption_annotations_file: JSON file containing caption annotations.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    """

    logging.info("writing to output path: %s", output_path)
    writers = [
        tf.python_io.TFRecordWriter(
            output_path + "-%05d-of-%05d.tfrecord" % (i, num_shards)
        )
        for i in range(num_shards)
    ]
    if images_info_file:
        images = _load_images_info(images_info_file)
    else:
        images = list(_load_images_info_from_dir(image_dir))

    img_to_obj_annotation = None
    img_to_caption_annotation = None
    category_index = None
    num_attributes = None
    if object_annotations_file:
        (
            img_to_obj_annotation,
            category_index,
            num_attributes,
        ) = _load_object_annotations(object_annotations_file)
    if caption_annotations_file:
        img_to_caption_annotation = _load_caption_annotations(caption_annotations_file)

    def _get_object_annotation(image_id):
        if img_to_obj_annotation:
            return img_to_obj_annotation[image_id]
        else:
            return None

    def _get_caption_annotation(image_id):
        if img_to_caption_annotation:
            return img_to_caption_annotation[image_id]
        else:
            return None

    pool = multiprocessing.Pool()
    total_num_annotations_skipped = 0
    for idx, (_, tf_example, num_annotations_skipped) in enumerate(
        pool.imap(
            _pool_create_tf_example,
            [
                (
                    image,
                    image_dir,
                    _get_object_annotation(image["id"]),
                    category_index,
                    _get_caption_annotation(image["id"]),
                    include_masks,
                    num_attributes,
                )
                for image in images
            ],
        )
    ):
        if idx % 100 == 0:
            logging.info("On image %d of %d", idx, len(images))

        total_num_annotations_skipped += num_annotations_skipped
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()

    logging.info(
        "Finished writing, skipped %d annotations.", total_num_annotations_skipped
    )


def _get_binary_mask(encoded_pixels: List[int], height: int, width: int):
    """Converts RLE to a binary mask."""
    mask = np.zeros(height * width, dtype=np.uint8)
    for start_pixel, num_pixels in zip(encoded_pixels[::2], encoded_pixels[1::2]):
        start_pixel -= 1
        mask[start_pixel : start_pixel + num_pixels] = 1

    mask = mask.reshape((height, width), order="F")

    return mask


def main(_):
    assert FLAGS.image_dir, "`image_dir` missing."
    if FLAGS.image_info_file:
        images_info_file = FLAGS.image_info_file
    elif FLAGS.object_annotations_file:
        images_info_file = FLAGS.object_annotations_file
    elif FLAGS.caption_annotations_file:
        images_info_file = FLAGS.caption_annotations_file
    else:
        images_info_file = None
        logger.info(
            "`images_info_file` is None, so images_info is get from actual images from `images_dir`."
        )

    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.gfile.IsDirectory(directory):
        tf.gfile.MakeDirs(directory)

    _create_tf_record_from_coco_annotations(
        images_info_file,
        FLAGS.image_dir,
        FLAGS.output_file_prefix,
        FLAGS.num_shards,
        FLAGS.object_annotations_file,
        FLAGS.caption_annotations_file,
        FLAGS.include_masks,
    )


if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    app.run(main)
