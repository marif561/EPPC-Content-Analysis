#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def load(messages_path: Path, annotations_path: Path):
    msg = pd.read_csv(
        messages_path,
        usecols=["message_id", "TO_PAT_YN", "num_eppc_annotations", "has_eppc"],
        encoding="utf-8-sig",
    )
    ann = pd.read_csv(
        annotations_path,
        usecols=["message_id", "annotation_id", "TO_PAT_YN", "code", "subcode"],
        dtype={"code": str, "subcode": str},
        encoding="utf-8-sig",
    )
    return msg, ann


def overall_statistics(msg: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    n = msg["num_eppc_annotations"]
    rows = [
        ("Total messages", len(msg)),
        ("Messages with EPPC", int(msg["has_eppc"].sum())),
        ("Total EPPC annotations", len(ann)),
        ("Unique EPPC codes", ann["code"].nunique()),
        ("Unique EPPC subcodes", ann["subcode"].nunique()),
        ("Mean EPPC annotations/message", n.mean()),
        ("SD EPPC annotations/message", n.std()),
        ("Median EPPC annotations/message", n.median()),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def code_frequency(msg: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    out = (
        ann.groupby("code")
        .agg(EPPC_occurrences=("annotation_id", "size"),
             unique_messages=("message_id", "nunique"))
        .reset_index()
    )
    out["percentage_of_all_annotations"] = out["EPPC_occurrences"] / len(ann) * 100
    out["percentage_of_messages"] = out["unique_messages"] / len(msg) * 100
    return out.sort_values(["EPPC_occurrences", "code"], ascending=[False, True])


def subcode_frequency(ann: pd.DataFrame) -> pd.DataFrame:
    out = (
        ann.groupby(["code", "subcode"])
        .agg(EPPC_occurrences=("annotation_id", "size"),
             unique_messages=("message_id", "nunique"))
        .reset_index()
    )
    out["percentage_of_all_annotations"] = out["EPPC_occurrences"] / len(ann) * 100
    return out.sort_values(["EPPC_occurrences", "code", "subcode"],
                           ascending=[False, True, True])


def intensity(msg: pd.DataFrame) -> pd.DataFrame:
    g = msg.groupby("TO_PAT_YN")["num_eppc_annotations"]
    return pd.DataFrame({
        "messages": g.size(),
        "total_eppc_annotations": g.sum(),
        "mean_eppc_per_message": g.mean(),
        "sd_eppc_per_message": g.std(),
        "median_eppc_per_message": g.median(),
        "min_eppc_per_message": g.min(),
        "max_eppc_per_message": g.max(),
    }).reset_index()


def patient_provider(ann: pd.DataFrame) -> pd.DataFrame:
    out = (
        ann.groupby(["TO_PAT_YN", "code"])
        .agg(occurrences=("annotation_id", "size"),
             unique_messages=("message_id", "nunique"))
        .reset_index()
    )
    totals = ann.groupby("TO_PAT_YN").size()
    out["percentage_within_direction"] = (
        out["occurrences"] / out["TO_PAT_YN"].map(totals) * 100
    )
    return out.sort_values(["TO_PAT_YN", "code"])


def main():
    p = argparse.ArgumentParser(description="EPPC Aim 2a summary tables")
    p.add_argument("--messages", type=Path, default=Path("data/EPPC_Aim2a_Message_Level.csv"))
    p.add_argument("--annotations", type=Path, default=Path("data/EPPC_Aim2a_Annotation_Level.csv"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    msg, ann = load(args.messages, args.annotations)
    args.out.mkdir(parents=True, exist_ok=True)

    tables = {
        "EPPC_Aim2a_Overall_Statistics.csv": overall_statistics(msg, ann),
        "EPPC_Aim2a_Code_Frequency.csv": code_frequency(msg, ann),
        "EPPC_Aim2a_Subcode_Frequency.csv": subcode_frequency(ann),
        "EPPC_Aim2a_Intensity.csv": intensity(msg),
        "EPPC_Aim2a_Patient_Provider.csv": patient_provider(ann),
    }
    for name, df in tables.items():
        df.to_csv(args.out / name, index=False)
        print(f"wrote {args.out / name}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
