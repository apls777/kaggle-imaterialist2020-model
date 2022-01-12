from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from configs import factory as config_factory
from dataloader import mode_keys
from hyperparameters import params_dict
from modeling import factory as model_factory
from utils import input_utils

from kaggle_imaterialist2020_model.json_logger import get_logger
from kaggle_imaterialist2020_model.transforms import (
    convert_predictions_to_coco_annotations,
)
from kaggle_imaterialist2020_model.types import COCOAnnotation, Prediction

logger = get_logger(__name__)


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

    def segment(
        self,
        imgs: list[np.ndarray],
        filenames: list[str],
        min_score_threshold: float = 0.05,
    ) -> tuple[COCOAnnotation, ...]:
        # imgs: [#images * {0, ..., 255}^(H, W, RGB) * #images]
        imgs, num_dummies = self._pad_with_dummy(imgs)
        imgs, image_info = list(zip(*[self._preprocess(img) for img in imgs]))
        imgs = tf.stack(imgs)
        image_info = tf.stack(image_info)

        preds = self._segment(imgs=imgs, image_info=image_info)

        preds = {k: v.numpy() for k, v in preds.items()}
        preds = self._split_into_single_examples(preds)
        preds = self._trim_dummy(preds, num_dummies)
        anns = self._to_coco_annotations(
            preds, filenames, min_score_threshold=min_score_threshold
        )

        return tuple(anns)

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

    def _split_into_single_examples(self, d: Prediction) -> list[Prediction]:
        predictions = []
        for k, b in d.items():
            for i, v in enumerate(b):
                try:
                    predictions[i][k] = v
                except IndexError:
                    predictions.append({k: v})
        return predictions

    def _trim_dummy(
        self, predictions: list[Prediction], num_dummies: int
    ) -> list[Prediction]:
        return predictions[: len(predictions) - num_dummies]

    def _to_coco_annotations(
        self,
        predictions: list[Prediction],
        filenames: list[str],
        min_score_threshold: float = 0.05,
    ) -> list[COCOAnnotation]:
        anns = []
        for fn, pred in zip(filenames, predictions):
            anns += convert_predictions_to_coco_annotations(
                pred,
                filename=fn,
                image_id=0,
                score_threshold=min_score_threshold,
            )
        return anns
