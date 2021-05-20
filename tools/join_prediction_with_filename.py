import io
from pathlib import Path

import typer
import pandas as pd


def join_prediction_with_filename(
    prediction_json: Path = typer.Option(...),
    filename_csv: Path = typer.Option(...),
    out_path: Path = typer.Option(None, "-o"),
) -> None:
    df_pred = pd.read_json(prediction_json)
    df_fn = pd.read_csv(filename_csv)
    df = df_pred.merge(df_fn, on="image_id")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(out_path, orient="records", lines=True)
    else:
        buf = io.StringIO()
        df.to_json(buf, orient="records", lines=True)
        print(buf.getvalue())


def main():
    typer.run(join_prediction_with_filename)


if __name__ == "__main__":
    main()
