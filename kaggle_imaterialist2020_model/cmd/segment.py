from __future__ import annotations

import tensorflow as tf
import typer
from kaggle_imaterialist2020_model.counter import Counter
from kaggle_imaterialist2020_model.io import Reader, Writer
from kaggle_imaterialist2020_model.json_logger import get_logger
from kaggle_imaterialist2020_model.segmentor import Segmentor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = get_logger(__name__)


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
    image_dir: str = typer.Option(
        ...,
        help="The path where there are images to segment. "
        "Choose from GCS URI (gs://bucket/images) "
        "or local path (path/to/images).",
    ),
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
        anns = segmentor.segment(
            imgs,
            filenames,
            min_score_threshold=min_score_threshold,
        )

        writer.write(anns)
        counter.count_success(len(imgs))
        counter.count_processed(len(imgs))
        counter.log_progress(logger.info)


if __name__ == "__main__":
    typer.run(main)
