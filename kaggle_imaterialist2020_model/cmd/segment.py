from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NewType

import numpy as np
import tensorflow as tf
import typer
from configs import factory as config_factory
from dataloader import mode_keys
from evaluation.submission import get_new_image_size
from hyperparameters import params_dict
from kaggle_imaterialist2020_model.counter import Counter
from kaggle_imaterialist2020_model.io import Reader, Writer
from kaggle_imaterialist2020_model.json_logger import get_logger
from modeling import factory as model_factory
from pycocotools import mask as mask_api
from typing_extensions import TypedDict
from utils import box_utils, input_utils, mask_utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = get_logger(__name__)

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


# TODO: 接頭辞の Bbox を取るために、
# COCOAnnotation や convert_pred... を別ファイルに移す。
# Bbox を付けてる理由は、 COCORLE の Width, Height と衝突するから。
BboxLeft = NewType("BboxLeft", float)
BboxTop = NewType("BboxTop", float)
BboxWidth = NewType("BboxWidth", float)
BboxHeight = NewType("BboxHeight", float)


class COCOAnnotation(TypedDict):
    image_id: int
    filename: str
    category_id: int
    # Avoid `bbox: list[float]` because
    # it's hard to know what each dimension means.
    # Also avoid `dict` like `{"left", "top", "width", "heiht"}`
    # along with the official COCO schema,
    # which adopts `list` instead of `dict`.
    bbox: tuple[BboxLeft, BboxTop, BboxWidth, BboxHeight]
    mask_area_fraction: float
    score: float
    segmentation: COCORLE
    mask_mean_score: float


class Segmentor:
    def __init__(
        self,
        config_file: str,
        checkpoint_path: str,
        batch_size: int,
        resize_shape: tuple[int, int],
        cache_dir: str,
        device: int | None = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.resize_shape = resize_shape

        params = config_factory.config_generator("mask_rcnn")
        if config_file:
            params = params_dict.override_params_dict(
                params, config_file, is_strict=True
            )
        params.validate()
        params.lock()

        self._model = model_factory.model_generator(params)
        estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
        )

        # TODO: write why not to use Estimator.predict()
        # because it is not able to download images from GCS

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir_parent = cache_dir or tmpdir
            children = list(Path(export_dir_parent).glob("*"))
            if children == []:
                logger.info(f"export saved_model: {export_dir_parent}")
                estimator.export_saved_model(
                    export_dir_base=export_dir_parent,
                    serving_input_receiver_fn=self._serving_input_receiver_fn,
                    checkpoint_path=checkpoint_path,
                )

            children = list(Path(export_dir_parent).glob("*"))
            export_dir = str(children[0])
            logger.info(f"load saved_model from {export_dir}")
            self.saved_model = tf.saved_model.load(export_dir=export_dir)

    def segment(self, imgs: list[np.ndarray]) -> tuple[Prediction, ...]:
        # imgs: [#images * {0, ..., 255}^(H, W, RGB) * #images]
        imgs, num_dummies = self._pad_with_dummy(imgs)
        imgs, image_info = list(zip(*[self._preprocess(img) for img in imgs]))
        imgs = tf.stack(imgs)
        image_info = tf.stack(image_info)

        preds = self._segment(imgs=imgs, image_info=image_info)

        preds = {k: v.numpy() for k, v in preds.items()}
        preds = self._split_into_single_examples(preds)
        preds = self._trim_dummy(preds, num_dummies)

        return preds

    def _pad_with_dummy(self, imgs: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        # pad the last batch with dummy images for fixed batch_size
        # imgs: [#images * {0, ..., 255}^(H, W, RGB) * #images]
        dummy = np.zeros((640, 640, 3))

        num_dummies = self.batch_size - len(imgs)
        imgs = imgs + [dummy for _ in range(num_dummies)]
        return imgs, num_dummies

    def _preprocess(self, image: np.ndarray) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.convert_to_tensor(image, tf.uint8)
        image = input_utils.normalize_image(image)
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self.resize_shape,
            self.resize_shape,
            aug_scale_min=1.0,
            aug_scale_max=1.0,
        )
        # image_info: (4, 2)
        # [
        #     [original_height, original_width],
        #     [desired_height,  desired_width ],
        #     [y_scale,         x_scale       ],
        #     [y_offset,        x_offset      ]
        # ]
        image.set_shape([self.resize_shape[0], self.resize_shape[1], 3])
        return image, image_info

    def _segment(self, imgs: tf.Tensor, image_info: tf.Tensor) -> Prediction:
        with tf.device(
            "/device:cpu:0" if self.device is None else f"/device:gpu:{self.device}"
        ):
            preds = self.saved_model.signatures["serving_default"](
                images=imgs,
                image_info=image_info,
            )
        return preds

    def _serving_input_receiver_fn(
        self,
    ) -> tf.estimator.export.ServingInputReceiver:
        images = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, self.resize_shape[0], self.resize_shape[1], 3],
            name="images",
        )
        image_info = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, 4, 2],
            name="image_info",
        )
        # a single image_info: (4, 2)
        # [
        #     [original_height, original_width],
        #     [desired_height,  desired_width ],
        #     [y_scale,         x_scale       ],
        #     [y_offset,        x_offset      ]
        # ]

        # Don't do this:
        #     receiver_tensors = {"images": images, "labels": {"image_info": image_info}} # noqa: E501
        # because the nesting depth of `features` must be 1
        # when exporting a SavedModel.
        receiver_tensors = {"images": images, "image_info": image_info}

        return tf.estimator.export.ServingInputReceiver(
            features=receiver_tensors, receiver_tensors=receiver_tensors
        )

    def _model_fn(self, features, labels, mode, params) -> tf.estimator.EstimatorSpec:
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
        labels = {"image_info": features["image_info"]}

        outputs = self._model.build_outputs(images, labels, mode=mode_keys.PREDICT)

        predictions = self._model.build_predictions(outputs, labels)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions
        )

    def _split_into_single_examples(self, d: Prediction) -> tuple[Prediction]:
        predictions = []
        for k, b in d.items():
            for i, v in enumerate(b):
                try:
                    predictions[i][k] = v
                except IndexError:
                    predictions.append({k: v})
        return tuple(predictions)

    def _trim_dummy(
        self, predictions: tuple[Prediction], num_dummies: int
    ) -> tuple[Prediction]:
        return predictions[: len(predictions) - num_dummies]


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
    """This is made, modifying a function of the same name in
    /tf_tpu_models/official/detection/evaluation/coco_utils.py

    Parameters
    ----------
    prediction : Prediction
        [description]
    output_image_size : int, optional
        [description], by default None
    score_threshold : float, optional
        [description], by default 0.05

    Returns
    -------
    list[COCOAnnotation]
        [description]
    """
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
            # Avoid `astype(np.float32)` because
            # it can't be serialized as JSON.
            "bbox": tuple(
                float(x) for x in prediction["pred_detection_boxes"][k] / eval_scale
            ),
            "mask_area_fraction": float(mask_area_fractions[m]),
            "score": float(prediction["pred_detection_scores"][k]),
            "segmentation": encoded_masks[m],
            "mask_mean_score": mask_mean_scores[m],
        }
        coco_annotations.append(ann)

    return coco_annotations


def main(
    config_file: str = typer.Option(
        ...,
        help="A config YAML file to load a trained model. "
        "Choose from GCS URI (gs://bucket/models/foo/config.yaml) "
        "or local path (path/to/config.yaml).",
    ),
    checkpoint_path: str = typer.Option(
        ...,
        help="A Tensorflow checkpoint file to load a trained model. "
        "Choose from GCS URI (gs://bucket/models/foo/model.ckpt-1234) "
        "or local path (path/to/model.ckpt-1234).",
    ),
    image_dir: str = typer.Option(...),
    gcs_project: str = typer.Option(
        None,
        help="The GCP Project where the bucket exists, which is given as `img_dir`.",
    ),
    cache_dir: str = typer.Option(
        None, help="a directory path to cache a Saved Model for efficient debugging."
    ),
    out: str = typer.Option(
        None,
        help="Where to save results. "
        "Choose from BQ table (bq://project.dataset.table) "
        "or local path (/path/to/segmentation.jsonlines).",
    ),
    batch_size: int = 2,
    image_size: int = 640,
    min_score_threshold: float = 0.05,
):
    reader = Reader(image_dir, gcs_project)
    writer = Writer(out=out)

    logger.info("segment")
    counter = Counter(total=len(list(reader.list_image_refs())))

    segmentor = Segmentor(
        config_file=config_file,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        resize_shape=(image_size, image_size),
        device=None,
        cache_dir=cache_dir,
    )

    for filenames, imgs in reader.read_image_batches(batch_size):
        preds = segmentor.segment(imgs)

        for fn, p in zip(filenames, preds):
            p["filename"] = fn
            p["pred_source_id"] = 0

            anns = convert_predictions_to_coco_annotations(
                p,
                score_threshold=min_score_threshold,
            )

            writer.write(anns)
            counter.count_success(1)
            counter.count_processed(1)
        counter.log_progress(logger.info)


if __name__ == "__main__":
    typer.run(main)
