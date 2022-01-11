from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NewType, Any

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from kaggle_imaterialist2020_model.cmd.segment import main as segment
from segmentation.transforms import coco_rle_to_mask

resource_dir = Path(__file__).parents[2] / "tests/resources"
image_dir = resource_dir / "images"
mask_dir = resource_dir / "masks"

Height = NewType("Height", int)
Width = NewType("Width", int)


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    rle["counts"] = rle["counts"].lstrip("b").strip("'").replace("\\\\", "\\").encode()
    mask = coco_rle_to_mask(rle)
    # {0, 1}^(heght, width)
    return mask


def iou(a: np.ndarray, e: np.ndarray) -> np.float64:
    return np.logical_and(a, e).sum() / np.logical_or(a, e).sum()


IMAGE = np.array(Image.open(image_dir / "00a8764cff12b2e849c850f4be5608bc.jpg"))


def save_mask_image(mask, out):
    t = 0.7
    plt.imshow(((1 - t) * IMAGE + t * 255 * mask[:, :, None]).astype(np.uint8))
    plt.axis("off")
    plt.savefig(out, bbox_inches="tight")


def join_cateogry(df):
    # TOOD: Include the following categories into training artifacts
    # and dynamicaly load them from the config at this script.
    # pasted from https://github.com/hrsma2i/dataset-iMaterialist/blob/main/raw/classes.txt
    df_c = (
        pd.Series(
            [
                "background",
                "shirt|blouse",
                "top|t-shirt|sweatshirt",
                "sweater",
                "cardigan",
                "jacket",
                "vest",
                "pants",
                "shorts",
                "skirt",
                "coat",
                "dress",
                "jumpsuit",
                "cape",
                "glasses",
                "hat",
                "headband|head_covering|hair_accessory",
                "tie",
                "glove",
                "watch",
                "belt",
                "leg_warmer",
                "tights|stockings",
                "sock",
                "shoe",
                "bag|wallet",
                "scarf",
                "umbrella",
            ]
        )
        .reset_index()
        .rename(columns={"index": "category_id", 0: "category"})
    )

    df = df.merge(df_c, on="category_id")

    return df


def crop_and_resize(mask):
    # TODO: Remove this function after fixing the mask resizing bug
    # https://github.com/hrsma2i/kaggle-imaterialist2020-model/pull/11
    h, w, _ = IMAGE.shape
    h_ = 640
    w_ = int(640 / h * w)
    resized_mask = cv2.resize(mask[:h_, :w_], (w, h))
    return resized_mask


def _partialy_reverse(mask, scale):
    ys, xs = np.where(mask == 1)
    top = ys.min()
    bottom = ys.max()
    height = int(scale * (bottom - top))
    left = xs.min()
    right = xs.max()
    width = int(scale * (right - left))
    mask[top : top + height, left : left + width] = (
        1 - mask[top : top + height, left : left + width]
    )
    return mask


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
    out_qual: Path = typer.Option(
        None,
        help="The path to save images for qualitative evaluation.",
    ),
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
        segment(
            config_file=config_file,
            checkpoint_path=checkpoint_path,
            image_dir=str(image_dir),
            out=f.name,
        )

        print("load segmentation")
        df = pd.read_json(f.name, lines=True)
        df = join_cateogry(df)
        actual_masks = df["segmentation"].apply(rle_to_mask)
        # TODO: Remove mask cropping & resizing after fixing the mask resizing bug
        # https://github.com/hrsma2i/kaggle-imaterialist2020-model/pull/11
        actual_masks = actual_masks.apply(crop_and_resize)

        if out_qual:
            print(f"save actual mask images at: {out_qual}")
            out_qual.mkdir(parents=True, exist_ok=True)
            df["actual_mask"] = actual_masks
            df.reset_index().apply(
                lambda row: save_mask_image(
                    row["actual_mask"],
                    out_qual / f"actual_{row['index']}_{row['category']}.png",
                ),
                axis=1,
            )

        print(f"check each expected mask exists in the actual masks")
        for mask_file in mask_dir.glob("*.npy"):

            expected = np.load(mask_file)
            # TODO: Remove mask cropping & resizing after fixing the mask resizing bug
            # https://github.com/hrsma2i/kaggle-imaterialist2020-model/pull/11
            expected = crop_and_resize(expected)

            expected = _partialy_reverse(expected, scale=0.2)

            if out_qual:
                save_mask_image(expected, out_qual / f"expected_{mask_file.stem}.png")

            assert actual_masks.apply(
                lambda actual: iou(actual, expected) > 0.90
            ).any(), f"{mask_file.name} mask doesn't exist in the prediction."

            print(f"{mask_file}: OK")


if __name__ == "__main__":
    typer.run(main)
