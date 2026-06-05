"""Batch analytics and trend detection for EngageIQ."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_generator import DATA_PATH, ensure_snapshot


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    ensure_snapshot(path)
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df


def summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "records": int(len(df)),
        "sources": sorted(df["source"].dropna().unique().tolist()),
        "domains": int(df["domain"].nunique()),
        "avg_activity": round(float(df["activity"].mean()), 3),
        "max_growth": round(float(df["growth_rate"].max()), 3),
    }


def trends(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    by_domain = (
        df.groupby("domain")
        .agg(records=("id", "count"), avg_growth=("growth_rate", "mean"), avg_activity=("activity", "mean"))
        .reset_index()
        .sort_values(["avg_growth", "records"], ascending=False)
        .head(15)
    )
    by_source = df.groupby("source").size().reset_index(name="records").sort_values("records", ascending=False)
    by_community = (
        df.groupby(["source", "community"])
        .agg(records=("id", "count"), avg_comments=("comments", "mean"), avg_growth=("growth_rate", "mean"))
        .reset_index()
        .sort_values(["avg_growth", "records"], ascending=False)
        .head(12)
    )
    daily = (
        df.assign(day=df["created_at"].dt.date)
        .groupby("day")
        .size()
        .reset_index(name="records")
        .sort_values("day")
        .tail(30)
    )
    return {
        "domains": to_records(by_domain),
        "sources": to_records(by_source),
        "communities": to_records(by_community),
        "daily": to_records(daily),
    }


def to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

