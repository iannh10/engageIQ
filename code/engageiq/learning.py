"""Adaptive feedback simulation."""

from __future__ import annotations

import random

import pandas as pd

from .ranking import OpportunityRanker, UserProfile, split_tags


def simulate_feedback(ranker: OpportunityRanker, profile: UserProfile, rounds: int = 60) -> dict[str, object]:
    """Run simulated engage/skip feedback and report ranking improvement.

    The simulator treats items sharing persona-interest tokens as relevant. It is
    intentionally simple and explainable for the final demo.
    """

    rng = random.Random(423)
    interest_tokens = {t for t in profile.interests.lower().replace("/", " ").replace(",", " ").split() if len(t) > 3}
    before = ranker.recommend(profile, limit=20)
    before_precision = precision_at_k(before, interest_tokens, 10)

    actions = {"engage": 0, "bookmark": 0, "skip": 0}
    for _ in range(rounds):
        recs = ranker.recommend(profile, limit=8)
        if recs.empty:
            break
        row = recs.sample(1, random_state=rng.randint(1, 100000)).iloc[0]
        tags = set(split_tags(row.get("tags", "")))
        text = f"{row.get('title', '')} {row.get('body', '')}".lower()
        relevant = bool(tags & interest_tokens) or any(token in text for token in interest_tokens)
        action = "engage" if relevant and rng.random() < 0.65 else "bookmark" if relevant else "skip"
        ranker.update_feedback(row, action)
        actions[action] += 1

    after = ranker.recommend(profile, limit=20)
    after_precision = precision_at_k(after, interest_tokens, 10)
    return {
        "rounds": rounds,
        "actions": actions,
        "precision_at_10_before": round(before_precision, 3),
        "precision_at_10_after": round(after_precision, 3),
        "improvement": round(after_precision - before_precision, 3),
    }


def precision_at_k(frame: pd.DataFrame, interest_tokens: set[str], k: int = 10) -> float:
    if frame.empty:
        return 0.0
    hits = 0
    for _, row in frame.head(k).iterrows():
        text = f"{row.get('title', '')} {row.get('body', '')} {row.get('tags', '')}".lower()
        if any(token in text for token in interest_tokens):
            hits += 1
    return hits / min(k, len(frame))
