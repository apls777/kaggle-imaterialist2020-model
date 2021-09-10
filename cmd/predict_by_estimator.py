from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NewType

import numpy as np
import tensorflow_core._api.v1.compat.v1 as tf
import typer
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

    def __repr__(self):
        return f"image_id: {self.image_id}, filename: {self.filename}, category: {self.imat_category_id}"


DUMMY_FILENAME = "DUMMY_FILENAME"


def load_image(path: str) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    return image


def load_and_preprocess_image(path: str, image_size: int):
    """[summary]

    Parameters
    ----------
    path : str
        image file path or GCS URI.
    image_size : int
        the image will be resized to (image_size, image_size, 3)

    Returns
    -------
    image_info: tf.Tensor
        dtype = tf.float32
        shape = (original|desired|scale|offset=4, height(y)|width(x)=2)

    c.f.,
        tf_tpu_models/official/detection/main.py @ FLAGS.mode == "predict"
        -> tf_tpu_models/official/detection/dataloader/input_reader.InputFn._parser_fn
        -> dataloader.maskrcnn_parser.Parser._parse_predict_data
            (parse loaded TF Record data to image and labels)

    image: tf.Tensor
        dtype = tf.float32
        shape = (height, width, RGB=3)
        image tensor that is preproessed to have normalized value and

    labels: dict[str, tf.Tensor]
        a dictionary of tensors used for training. The following
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

    # pad the last batch with dummy images to fix batch_size
    dummy_image = tf.zeros([image_size, image_size, 3], dtype=tf.uint8)
    dummy_image.set_shape([None, None, 3])

    image = tf.cond(
        tf.math.equal(path, DUMMY_FILENAME),
        lambda: dummy_image,
        lambda: load_image(path),
    )

    image = input_utils.normalize_image(image)
    resize_shape = [image_size, image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image, resize_shape, resize_shape, aug_scale_min=1.0, aug_scale_max=1.0
    )
    image.set_shape([resize_shape[0], resize_shape[1], 3])

    labels = {"image_info": image_info}

    feature = {"images": image, "labels": labels}

    return feature


class InputFn:
    def __init__(self, filenames: str, batch_size: int, image_size: int):
        self.filenames = filenames
        self.batch_size = batch_size
        self.image_size = image_size

    def __call__(self):
        print("FILENAMES: ", self.filenames)
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
            tf_tpu_models/official/detection/main.py @ model_fn = model_builder.ModelFn(params)
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


def main(
    config_file: str = typer.Option(...),
    checkpoint_path: str = typer.Option(...),
    image_dir: str = typer.Option(...),
    batch_size: int = 2,
    image_size: int = 640,
    min_score_threshold: float = 0.05,
):
    params = config_factory.config_generator("mask_rcnn")
    if config_file:
        params = params_dict.override_params_dict(params, config_file, is_strict=True)
    params.validate()
    params.lock()

    estimator = tf.estimator.Estimator(
        model_fn=ModelFn(params),
    )

    image_files_pattern = f"{image_dir.rstrip('/')}/*"
    filenames: list[str] = tf.io.gfile.glob(image_files_pattern)

    predictor = estimator.predict(
        input_fn=InputFn(
            filenames=filenames,
            batch_size=batch_size,
            image_size=image_size,
        ),
        checkpoint_path=checkpoint_path,
        yield_single_examples=False,
    )

    for b, predictions in enumerate(predictor):
        # c.f., tf_tpu_models/official/detection/executor/tpu_executor.TPUExecutor.predict
        predictions_ = {k.replace("pred_", ""): [v] for k, v in predictions.items()}

        # Add "source_id" because `labels` doesn't have ["groundtruth"]["source_id"]
        source_ids = [
            np.array([filenames.index(fn) for fn in filenames[b * batch_size : (b+1) * batch_size]])
        ]
        predictions_["source_id"] = source_ids
        # shape=(num_batches, batch_size, )

        coco_annotations = coco_utils.convert_predictions_to_coco_annotations(
            predictions_,
            output_image_size=1024,
            score_threshold=min_score_threshold,
        )

        for ann in coco_annotations:
            source_id = ann["image_id"]
            filename = os.path.basename(filenames[source_id])
            if filename != DUMMY_FILENAME:
                # In the case of byte type, it cannot be converted to json
                ann["segmentation"]["counts"] = str(ann["segmentation"]["counts"])

                seg = Segmentation(
                    image_id=source_id,
                    filename=filename,
                    segmentation=ann["segmentation"],
                    imat_category_id=ann["category_id"],
                    score=ann["score"],
                    mask_mean_score=ann["mask_mean_score"],
                )
                print(seg)


if __name__ == "__main__":
    typer.run(main)
