from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.model_selection import GroupKFold


def create_group_folds(
    input_csv: str,
    output_csv: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["fold"] = -1
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (_, val_idx) in enumerate(splitter.split(df, groups=df["group"])):
        df.loc[val_idx, "fold"] = fold

    df["fold"] = df["fold"].astype(int)
    df.to_csv(output_csv, index=False)

    for fold in range(n_splits):
        train_groups = set(df.loc[df["fold"] != fold, "group"])
        val_groups = set(df.loc[df["fold"] == fold, "group"])
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Group leakage detected in fold {fold}: {overlap}")
        logging.info("Fold %s validation rows: %s", fold, int((df["fold"] == fold).sum()))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.csv")
    parser.add_argument("--output", default="dataset_folds.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    create_group_folds(args.input, args.output, args.n_splits)
    logging.info("Saved grouped folds to %s", args.output)


if __name__ == "__main__":
    main()
