from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import NewType

import numpy as np
import tensorflow_core._api.v1.compat.v1 as tf
import typer
from configs import factory as config_factory
from dataloader import mode_keys
from evaluation.submission import get_new_image_size
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField
from google.cloud.bigquery.enums import SqlTypeNames
from hyperparameters import params_dict
from modeling import factory as model_factory
from pycocotools import mask as mask_api
from typing_extensions import TypedDict
from utils import box_utils, input_utils, mask_utils

from counter import Counter

DUMMY_FILENAME = "DUMMY_FILENAME"


def load_image(path: str) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    return image


class Label(TypedDict):
    image_info: tf.Tensor  # (2, 2)=(original|scale, height, width)


class Feature(TypedDict):
    images: tf.Tensor  # (height, width, RGB=3)
    labels: Label


def load_and_preprocess_image(path: str, image_size: int) -> Feature:
    """
    c.f.,
        tf_tpu_models/official/detection/main.py @ FLAGS.mode == "predict"
        -> tf_tpu_models/official/detection/dataloader/input_reader.InputFn._parser_fn
        -> dataloader.maskrcnn_parser.Parser._parse_predict_data
            (parse loaded TF Record data to image and labels)

    Parameters
    ----------
    path : str
        image file path or GCS URI.
    image_size : int
        the image will be resized to (image_size, image_size, 3)

    Returns
    -------
    Feature
    """
    # pad the last batch with dummy images to fix batch_size
    dummy_image = tf.zeros([image_size, image_size, 3], dtype=tf.uint8)
    dummy_image.set_shape([None, None, 3])

    image = tf.cond(
        tf.math.equal(path, DUMMY_FILENAME),
        lambda: dummy_image,
        lambda: load_image(path),
    )
    # tf.print(path, tf.shape(image))

    image = input_utils.normalize_image(image)
    resize_shape = [image_size, image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image, resize_shape, resize_shape, aug_scale_min=1.0, aug_scale_max=1.0
    )
    image.set_shape([resize_shape[0], resize_shape[1], 3])

    labels: Label = {"image_info": image_info}

    feature: Feature = {"images": image, "labels": labels}

    return feature


class InputFn:
    def __init__(self, filenames: list[str], batch_size: int, image_size: int):
        self.filenames = filenames
        self.batch_size = batch_size
        self.image_size = image_size

    def __call__(self):
        # pad the last batch with dummy images to fix batch_size
        self.filenames += [
            DUMMY_FILENAME for _ in range(len(self.filenames) % self.batch_size)
        ]
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset.map(
            lambda p: load_and_preprocess_image(p, image_size=self.image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # The last smaller batch is not actually dropped
        # because it is padded with dummy images.
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class ModelFn:
    def __init__(self, params):
        self._model = model_factory.model_generator(params)

    def __call__(self, features, labels, mode, params) -> tf.estimator.EstimatorSpec:
        """Returns a EstimatorSpec for prediction.

        c.f.,
            tf_tpu_models/official/detection/main.py @ model_fn = model_builder.ModelFn(params)  # noqa: E501
            > tf_tpu_models/official/detection/modeling/model_builder.ModelFn
            > tf_tpu_models/official/detection/modeling/base_model.BaseModel.predict

        Args:
        features: a dict of Tensors including the input images and other label
            tensors used for prediction.

        Returns:
        a EstimatorSpec object used for prediction.
        """
        # Include `labels` into `features`
        # because only `features` passed to this function
        # in Estimator.predict.
        images = features["images"]
        # images: (height, widht, RGB=3)
        labels = features["labels"]

        outputs = self._model.build_outputs(images, labels, mode=mode_keys.PREDICT)

        predictions = self._model.build_predictions(outputs, labels)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions
        )


Height = NewType("Height", int)
Width = NewType("Width", int)
RLE = NewType("RLE", str)


class COCORLE(TypedDict):
    size: tuple[Height, Width]
    counts: RLE


class Prediction(TypedDict):
    filename: str
    pred_source_id: int
    pred_num_detctions: int
    pred_image_info: np.array  # (2, 2)=(orginal|scale, height|width)  # noqa: E501
    pred_detection_boxes: np.array  # (num_detections, 4)
    pred_detection_classes: np.array  # (num_detections, )
    pred_detection_scores: np.array  # (num_detections, )
    pred_detection_masks: np.array  # (num_detections, mask_height, mask_width)


class COCOAnnotation(TypedDict):
    image_id: int
    filename: str
    category_id: int
    bbox: list[float]
    mask_area_fraction: float
    score: float
    segmentation: COCORLE
    mask_mean_score: float


def encode_mask_fn(x) -> COCORLE:
    encoded_mask = mask_api.encode(np.asfortranarray(x))
    # In the case of byte type, it cannot be converted to json
    encoded_mask["counts"] = str(encoded_mask["counts"])
    return encoded_mask


def convert_predictions_to_coco_annotations(
    prediction: Prediction,
    output_image_size: int = None,
    score_threshold=0.05,
) -> list[COCOAnnotation]:
    prediction["pred_detection_boxes"] = box_utils.yxyx_to_xywh(
        prediction["pred_detection_boxes"]
    )

    mask_boxes = prediction["pred_detection_boxes"]

    image_id = prediction["pred_source_id"]
    orig_image_size = prediction["pred_image_info"][0]
    # image_info: (2, 2)=(orginal|scale, height|width)  # noqa: E501

    if output_image_size:
        eval_image_size = get_new_image_size(orig_image_size, output_image_size)
    else:
        eval_image_size = orig_image_size

    eval_scale = orig_image_size[0] / eval_image_size[0]

    bbox_indices = np.argwhere(
        prediction["pred_detection_scores"] >= score_threshold
    ).flatten()

    predicted_masks = prediction["pred_detection_masks"][bbox_indices]
    image_masks = mask_utils.paste_instance_masks(
        predicted_masks,
        mask_boxes[bbox_indices].astype(np.float32) / eval_scale,
        int(eval_image_size[0]),
        int(eval_image_size[1]),
    )
    binary_masks = (image_masks > 0.0).astype(np.uint8)
    encoded_masks = [encode_mask_fn(binary_mask) for binary_mask in list(binary_masks)]

    mask_masks = (predicted_masks > 0.5).astype(np.float32)
    mask_areas = mask_masks.sum(axis=-1).sum(axis=-1)
    mask_area_fractions = (mask_areas / np.prod(predicted_masks.shape[1:])).tolist()
    mask_mean_scores = (
        (predicted_masks * mask_masks).sum(axis=-1).sum(axis=-1) / mask_areas
    ).tolist()

    coco_annotations: list[COCOAnnotation] = []
    for m, k in enumerate(bbox_indices):
        ann: COCOAnnotation
        ann = {
            "image_id": int(image_id),
            "filename": prediction["filename"],
            "category_id": int(prediction["pred_detection_classes"][k]),
            "bbox": (
                prediction["pred_detection_boxes"][k].astype(np.float32) / eval_scale
            ).tolist(),
            "mask_area_fraction": float(mask_area_fractions[m]),
            "score": float(prediction["pred_detection_scores"][k]),
            "segmentation": encoded_masks[m],
            "mask_mean_score": mask_mean_scores[m],
        }
        coco_annotations.append(ann)

    return coco_annotations


def insert_bq(
    bq_client: bigquery.Client,
    result_table: bigquery.Table | bigquery.TableReference | str,
    rows_to_insert: list[COCOAnnotation],
) -> None:

    errors = bq_client.insert_rows(result_table, rows_to_insert)  # Make an API request.
    if errors != []:
        print("Encountered errors while inserting rows: {}".format(errors))


def create_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
) -> bigquery.Table:
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    schema = [
        SchemaField("image_id", SqlTypeNames.INTEGER, mode="REQUIRED"),
        SchemaField("filename", SqlTypeNames.STRING, mode="REQUIRED"),
        SchemaField("category_id", SqlTypeNames.INTEGER, mode="REQUIRED"),
        SchemaField("score", SqlTypeNames.FLOAT, mode="REQUIRED"),
        SchemaField(
            "segmentation",
            SqlTypeNames.RECORD,
            mode="REQUIRED",
            fields=[
                SchemaField("size", SqlTypeNames.INTEGER, mode="REPEATED"),
                SchemaField("counts", SqlTypeNames.STRING, mode="REQUIRED"),
            ],
        ),
        SchemaField("bbox", SqlTypeNames.FLOAT, mode="REPEATED"),
        SchemaField("mask_area_fraction", SqlTypeNames.FLOAT, mode="REQUIRED"),
        SchemaField("mask_mean_score", SqlTypeNames.FLOAT, mode="REQUIRED"),
    ]

    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table, exists_ok=True)
    return table


class Destination(str, Enum):
    BQ = "bq"
    LOCAL = "local"


def main(
    config_file: str = typer.Option(
        ...,
        help="A config YAML file to load a trained model. "
        "Choose from GCS URI (gs://bucket/models/foo/config.yaml) or local path (path/to/config.yaml).",
    ),
    checkpoint_path: str = typer.Option(
        ...,
        help="A Tensorflow checkpoint file to load a trained model. "
        "Choose from GCS URI (gs://bucket/models/foo/model.ckpt-1234) or local path (path/to/model.ckpt-1234).",
    ),
    image_dir: str = typer.Option(...),
    out: str = typer.Option(
        None,
        help="Where to save results. "
        "Choose from BQ table (bq://project.dataset.table) or local path (/path/to/segmentation.jsonlines).",
    ),
    batch_size: int = 2,
    image_size: int = 640,
    min_score_threshold: float = 0.05,
):

    if out.startswith("bq://"):
        project_id, dataset_id, table_id = out.lstrip("bq://").split(".")
        bq_client = bigquery.Client(project=project_id)
        table = create_table(client=bq_client, dataset_id=dataset_id, table_id=table_id)
        dst = Destination.BQ
    else:
        out_json = Path(out)
        out_json.parent.mkdir(exist_ok=True, parents=True)
        out_file = out_json.open("w")
        dst = Destination.LOCAL

    params = config_factory.config_generator("mask_rcnn")
    if config_file:
        params = params_dict.override_params_dict(params, config_file, is_strict=True)
    params.validate()
    params.lock()

    estimator = tf.estimator.Estimator(
        model_fn=ModelFn(params),
    )

    image_files_pattern = f"{image_dir.rstrip('/')}/*"
    image_files: list[str] = tf.io.gfile.glob(image_files_pattern)

    predictor = estimator.predict(
        input_fn=InputFn(
            filenames=image_files,
            batch_size=batch_size,
            image_size=image_size,
        ),
        checkpoint_path=checkpoint_path,
        yield_single_examples=True,
    )

    counter = Counter(total=len(image_files))
    prediction: Prediction
    for source_id, prediction in enumerate(predictor):
        filename = os.path.basename(image_files[source_id])

        if filename != DUMMY_FILENAME:
            prediction["filename"] = os.path.basename(image_files[source_id])
            # Add "pred_source_id" because `labels` doesn't have ["groundtruth"]["source_id"]  # noqa: E501
            prediction["pred_source_id"] = source_id

            coco_annotations = convert_predictions_to_coco_annotations(
                prediction=prediction,
                # output_image_size=1024,
                score_threshold=min_score_threshold,
            )

            if dst == Destination.BQ:
                insert_bq(bq_client, table, coco_annotations)
            elif dst == Destination.LOCAL:
                out_file.write(
                    "\n".join([json.dumps(a) for a in coco_annotations]) + "\n"
                )

            counter.count_success(1)
            counter.count_processed(1)
            counter.log_progress()

    if dst == Destination.LOCAL:
        out_file.close()


if __name__ == "__main__":
    typer.run(main)
