from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from PIL import Image


def build_dataset_csv(train_dir: str | Path, output: str | Path) -> pd.DataFrame:
    train_dir = Path(train_dir)
    label_file = train_dir / "gcp_marks.json"
    with open(label_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    rows: list[dict] = []
    for rel_path, info in labels.items():
        full_path = train_dir / Path(rel_path)
        if not full_path.exists():
            logging.warning("Missing image: %s", rel_path)
            continue

        with Image.open(full_path) as image:
            width, height = image.size

        parts = Path(rel_path).parts
        if len(parts) < 4:
            logging.warning("Unexpected path structure: %s", rel_path)
            continue

        project, survey, gcp_id = parts[0], parts[1], parts[2]
        x = float(info["mark"]["x"])
        y = float(info["mark"]["y"])
        rows.append(
            {
                "path": Path(rel_path).as_posix(),
                "full_path": str(full_path),
                "project": project,
                "survey": survey,
                "gcp_id": gcp_id,
                "group": f"{project}/{survey}/{gcp_id}",
                "width": width,
                "height": height,
                "x": x,
                "y": y,
                "x_norm": x / width,
                "y_norm": y / height,
                "shape": info.get("verified_shape"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="train_dataset")
    parser.add_argument("--output", default="dataset.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = build_dataset_csv(args.train_dir, args.output)
    logging.info("Saved %s rows to %s", len(df), args.output)
    logging.info("Shape distribution:\n%s", df["shape"].value_counts(dropna=False))
    logging.info(
        "Projects=%s Surveys=%s GCP groups=%s",
        df["project"].nunique(),
        df["survey"].nunique(),
        df["group"].nunique(),
    )


if __name__ == "__main__":
    main()
