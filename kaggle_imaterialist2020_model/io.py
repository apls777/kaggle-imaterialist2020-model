from __future__ import annotations

import io
import json
from enum import Enum
from pathlib import Path
from typing import Generator
from google.cloud.storage.blob import Blob

import numpy as np
from google.cloud import bigquery, storage
from google.cloud.bigquery import SchemaField
from google.cloud.bigquery.enums import SqlTypeNames
from PIL import Image


class ImageSource(str, Enum):
    GCS = "gcs"
    LOCAL = "local"


class Reader:
    def __init__(self, img_dir: str, gcs_project: str):
        self.img_dir = img_dir
        if img_dir.startswith("gs://"):
            if gcs_project is not None:
                self.gcs_client = storage.Client(project=gcs_project)
            else:
                raise ValueError(
                    "Give `gcs_project` or give local image_dir instead of GCS URI."
                )
            self.img_src = ImageSource.GCS
        else:
            self.img_src = ImageSource.LOCAL

    def list_image_refs(self) -> Generator[Path | Blob]:
        if self.img_src == ImageSource.GCS:
            parts = self.img_dir.replace("gs://", "").rstrip("/").split("/")
            bucket = parts[0]
            prefix = "/".join(parts[1:])
            blobs = self.gcs_client.list_blobs(bucket, prefix=prefix)
            return blobs
        elif self.img_src == ImageSource.LOCAL:
            paths = Path(self.img_dir).glob("*")
            return paths

    def read_image_batches(
        self,
        batch_size: int,
    ) -> Generator[tuple[list[str], list[np.ndarray]]]:
        refs = list(self.list_image_refs())

        names = []
        imgs = []

        for i, ref in enumerate(refs):
            if self.img_src == ImageSource.GCS:
                blob: Blob = ref
                name = blob.name.split("/")[-1]
                buf = io.BytesIO(blob.download_as_bytes())
                img = self._buf_to_img(buf)
            elif self.img_src == ImageSource.LOCAL:
                path: Path = ref
                name = path.name
                img = self._buf_to_img(path)

            imgs.append(img)
            names.append(name)

            is_last = i == len(refs) - 1
            if len(imgs) == batch_size or is_last:
                yield names, imgs
                names = []
                imgs = []

    @staticmethod
    def _buf_to_img(buf) -> np.ndarray:
        img = np.array(Image.open(buf).convert("RGB"))
        return img


class Destination(str, Enum):
    BQ = "bq"
    LOCAL = "local"


class Writer:
    def __init__(self, out: str):
        if out.startswith("bq://"):
            project, dataset, table_id = out.lstrip("bq://").split(".")
            self.bq_client = bigquery.Client(project=project)
            self.table = self.create_table(dataset, table_id)
            self.dst = Destination.BQ
        else:
            self.out_json = Path(out)
            self.out_json.parent.mkdir(exist_ok=True, parents=True)
            self.dst = Destination.LOCAL

    def create_table(self, dataset: str, table_id: str) -> bigquery.Table:
        dataset_ref = self.bq_client.dataset(dataset)
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
        table = self.bq_client.create_table(table, exists_ok=True)
        return table

    def write(
        self,
        rows: list[dict],
    ) -> None:
        if self.dst == Destination.BQ:
            errors = self.bq_client.insert_rows(self.table, rows)
            if errors != []:
                raise Exception(errors[0])
        elif self.dst == Destination.LOCAL:
            with self.out_json.open("a") as f:
                f.write("\n".join([json.dumps(r) for r in rows]) + "\n")
