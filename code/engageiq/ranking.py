"""Embedding retrieval, scoring, re-ranking, and explanations."""

from __future__ import annotations

import math
import os
import json
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_generator import DATA_PATH, ensure_snapshot

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_CACHE_PATH = DATA_PATH.parent / "engageiq_sbert_embeddings.npy"
SBERT_META_PATH = DATA_PATH.parent / "engageiq_sbert_embeddings.json"

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - fallback for minimal graders
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dense embedding dependency
    SentenceTransformer = None


TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


@dataclass
class UserProfile:
    name: str
    interests: str
    goal: str
    platforms: list[str]
    time_budget: float
    avoid: str = ""

    @property
    def text(self) -> str:
        return f"{self.name}. {self.interests}. Goal: {self.goal}. Avoid: {self.avoid}"


class OpportunityRanker:
    def __init__(self, data_path: Path = DATA_PATH, embedding_backend: str | None = None) -> None:
        ensure_snapshot(data_path)
        self.data_path = data_path
        self.df = pd.read_csv(data_path, low_memory=False)
        self.df["search_text"] = (
            self.df["title"].fillna("")
            + " "
            + self.df["body"].fillna("")
            + " "
            + self.df["domain"].fillna("")
            + " "
            + self.df["tags"].fillna("")
        )
        self.feedback_weights: dict[str, float] = {}
        self.requested_backend = (embedding_backend or os.getenv("ENGAGEIQ_EMBEDDING_BACKEND", "sbert")).lower()
        self.embedding_backend = "uninitialized"
        self.embedding_note = ""
        self._fit_embeddings()

    def _fit_embeddings(self) -> None:
        if self.requested_backend in {"sbert", "dense", "auto"} and self._fit_sbert_embeddings():
            return
        self._fit_tfidf_embeddings()

    def _fit_sbert_embeddings(self) -> bool:
        if SentenceTransformer is None:
            self.embedding_note = "sentence-transformers is not installed; using TF-IDF fallback."
            return False
        try:
            model_name = os.getenv("ENGAGEIQ_SBERT_MODEL", SBERT_MODEL_NAME)
            local_only_env = os.getenv("ENGAGEIQ_SBERT_LOCAL_ONLY")
            local_only = SBERT_CACHE_PATH.exists() if local_only_env is None else local_only_env == "1"
            try:
                self.sbert_model = SentenceTransformer(model_name, local_files_only=local_only)
            except Exception:
                if local_only_env is None and local_only:
                    self.sbert_model = SentenceTransformer(model_name, local_files_only=False)
                else:
                    raise
            self.vectorizer = None
            self.matrix = self._load_or_build_sbert_cache(model_name)
            self.embedding_backend = "sbert"
            self.embedding_note = f"Dense SBERT embeddings active: {model_name}"
            return True
        except Exception as exc:
            self.embedding_note = f"SBERT unavailable ({exc}); using TF-IDF fallback."
            return False

    def _load_or_build_sbert_cache(self, model_name: str) -> np.ndarray:
        signature = dataset_signature(self.df)
        meta = read_json(SBERT_META_PATH)
        if (
            SBERT_CACHE_PATH.exists()
            and meta.get("model_name") == model_name
            and meta.get("signature") == signature
            and int(meta.get("rows", -1)) == len(self.df)
        ):
            return np.load(SBERT_CACHE_PATH)

        texts = self.df["search_text"].astype(str).tolist()
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=int(os.getenv("ENGAGEIQ_SBERT_BATCH_SIZE", "64")),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype("float32")
        np.save(SBERT_CACHE_PATH, embeddings)
        SBERT_META_PATH.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "rows": len(self.df),
                    "dimension": int(embeddings.shape[1]),
                    "signature": signature,
                },
                indent=2,
            )
        )
        return embeddings

    def _fit_tfidf_embeddings(self) -> None:
        if TfidfVectorizer is None:
            self.vectorizer = None
            self.matrix = None
            self.embedding_backend = "token_overlap"
            return
        self.vectorizer = TfidfVectorizer(
            max_features=6500,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])
        self.embedding_backend = "tfidf"
        if not self.embedding_note:
            self.embedding_note = "TF-IDF cosine retrieval active."

    def retrieve(self, profile: UserProfile, top_k: int = 250) -> pd.DataFrame:
        if self.embedding_backend == "sbert":
            query_vec = self.sbert_model.encode(
                [profile.text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype("float32")
            sims = self.matrix @ query_vec
            count = min(top_k, len(sims))
            idx = np.argpartition(sims, -count)[-count:]
            frame = self.df.iloc[idx].copy()
            frame["relevance"] = sims[idx]
            return frame.sort_values("relevance", ascending=False)
        if self.vectorizer is None or cosine_similarity is None:
            return self._retrieve_by_overlap(profile, top_k)
        query_vec = self.vectorizer.transform([profile.text])
        sims = cosine_similarity(query_vec, self.matrix).ravel()
        count = min(top_k, len(sims))
        idx = np.argpartition(sims, -count)[-count:]
        frame = self.df.iloc[idx].copy()
        frame["relevance"] = sims[idx]
        return frame.sort_values("relevance", ascending=False)

    def _retrieve_by_overlap(self, profile: UserProfile, top_k: int) -> pd.DataFrame:
        query_tokens = set(TOKEN_RE.findall(profile.text.lower()))
        scores = []
        for text in self.df["search_text"]:
            tokens = set(TOKEN_RE.findall(str(text).lower()))
            scores.append(len(query_tokens & tokens) / max(1, len(query_tokens)))
        frame = self.df.copy()
        frame["relevance"] = scores
        return frame.sort_values("relevance", ascending=False).head(top_k)

    def update_feedback(self, row: pd.Series, action: str) -> None:
        delta = {"engage": 0.08, "bookmark": 0.05, "skip": -0.07}.get(action, 0.0)
        for token in split_tags(row.get("tags", "")):
            self.feedback_weights[token] = self.feedback_weights.get(token, 0.0) + delta

    def recommend(self, profile: UserProfile, limit: int = 10) -> pd.DataFrame:
        candidates = self.retrieve(profile, top_k=500)
        candidates = candidates[candidates["source"].isin(profile.platforms)] if profile.platforms else candidates
        if profile.avoid:
            avoid = [x.strip().lower() for x in re.split(r"[,;]", profile.avoid) if len(x.strip()) > 3]
            if avoid:
                pattern = "|".join(re.escape(x) for x in avoid)
                filtered = candidates[~candidates["search_text"].str.lower().str.contains(pattern, na=False)]
                if not filtered.empty:
                    candidates = filtered
        scored = self.score(candidates, profile)
        reranked = self.diversify(scored, limit=limit)
        if reranked.empty:
            return reranked
        reranked["why_this"] = reranked.apply(lambda row: explain(row, profile), axis=1)
        reranked["suggested_action"] = reranked.apply(suggest_action, axis=1)
        return reranked

    def score(self, frame: pd.DataFrame, profile: UserProfile) -> pd.DataFrame:
        frame = frame.copy()
        max_rel = max(float(frame["relevance"].max()), 1e-6) if len(frame) else 1
        frame["relevance_norm"] = frame["relevance"] / max_rel
        frame["health"] = (
            0.45 * norm(frame["activity"]) + 0.25 * norm(frame["comments"]) + 0.20 * norm(frame["contributors"]) - 0.10 * norm(frame["toxicity"])
        ).clip(0, 1)
        frame["visibility"] = (
            0.45 * norm(frame["score"]) + 0.35 * norm(frame["stars"]) + 0.20 * norm(frame["growth_rate"])
        ).clip(0, 1)
        frame["freshness"] = frame["created_at"].apply(freshness_score)
        ideal_minutes = max(20, (profile.time_budget * 60) / 5)
        frame["effort_fit"] = 1 - (abs(frame["effort_minutes"] - ideal_minutes) / max(ideal_minutes * 2, 1))
        frame["effort_fit"] = frame["effort_fit"].clip(0, 1)
        frame["feedback_affinity"] = frame["tags"].apply(lambda tags: feedback_score(split_tags(tags), self.feedback_weights))
        frame["final_score"] = (
            45 * frame["relevance_norm"]
            + 15 * frame["health"]
            + 15 * frame["visibility"]
            + 10 * frame["freshness"]
            + 10 * frame["effort_fit"]
            + 5 * frame["feedback_affinity"]
        )
        if "trend" in profile.goal.lower() or "journalist" in profile.name.lower():
            frame["final_score"] += 10 * frame["growth_rate"] + 5 * frame["freshness"]
        if "beginner" in profile.interests.lower():
            frame["final_score"] += frame["good_first_issue"].astype(float) * 10
        if "devops" in profile.name.lower() or "kubernetes" in profile.interests.lower():
            frame["final_score"] += frame["domain"].isin(["DevOps/K8s", "Cloud APIs"]).astype(float) * 8
        if "startup" in profile.name.lower() or "developer productivity" in profile.interests.lower():
            startup_domains = ["Developer Tools", "Cloud APIs", "Trending Open-Source"]
            startup_terms = r"developer|productivity|api|sdk|cli|tool|open source|workflow|startup"
            frame["final_score"] += frame["domain"].isin(startup_domains).astype(float) * 10
            frame["final_score"] += frame["search_text"].str.lower().str.contains(startup_terms, na=False).astype(float) * 6
            weak_domains = ["GameDev (C++)", "Embedded Systems (C/RTOS)", "Mobile Dev (iOS/Flutter)"]
            frame["final_score"] -= frame["domain"].isin(weak_domains).astype(float) * 6
        return frame.sort_values("final_score", ascending=False)

    @staticmethod
    def diversify(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        selected = []
        domain_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for _, row in frame.iterrows():
            domain_penalty = max(0, domain_counts.get(row["domain"], 0) - 1) * 6
            source_penalty = max(0, source_counts.get(row["source"], 0) - 3) * 4
            adjusted = float(row["final_score"]) - domain_penalty - source_penalty
            row = row.copy()
            row["diversified_score"] = round(adjusted, 2)
            selected.append(row)
            domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            selected.sort(key=lambda x: x["diversified_score"], reverse=True)
            selected = selected[:limit]
        return pd.DataFrame(selected).sort_values("diversified_score", ascending=False).head(limit)


def norm(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def freshness_score(value: object) -> float:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        return max(0.0, 1.0 - age_hours / (24 * 35))
    except Exception:
        return 0.5


def split_tags(tags: object) -> list[str]:
    return [t.strip().lower() for t in str(tags).split(",") if t.strip()]


def feedback_score(tags: Iterable[str], weights: dict[str, float]) -> float:
    vals = [weights.get(tag, 0.0) for tag in tags]
    if not vals:
        return 0.5
    return max(0.0, min(1.0, 0.5 + sum(vals) / len(vals)))


def dataset_signature(frame: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(len(frame)).encode("utf-8"))
    for row_id, text in zip(frame["id"].astype(str), frame["search_text"].astype(str)):
        hasher.update(row_id.encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def explain(row: pd.Series, profile: UserProfile) -> str:
    pieces = [
        f"{row['domain']} match for {profile.name}",
        f"semantic relevance {float(row['relevance_norm']):.2f}",
        f"community health {float(row['health']):.2f}",
        f"visibility {float(row['visibility']):.2f}",
        f"estimated effort {int(row['effort_minutes'])} min",
    ]
    if int(row.get("good_first_issue", 0)):
        pieces.append("good-first-issue signal")
    if float(row.get("growth_rate", 0)) > 0.7:
        pieces.append("rising fast")
    return "; ".join(pieces) + "."


def suggest_action(row: pd.Series) -> str:
    source = row["source"]
    domain = row["domain"]
    if source in {"github", "gh_archive"}:
        if int(row.get("good_first_issue", 0)):
            return "Open the issue, comment with your intended approach, then submit a small PR."
        return "Review open issues, add a focused technical comment, and propose a small contribution path."
    if source == "reddit":
        return f"Write a concise comment sharing a concrete {domain} example and ask one follow-up question."
    return f"Draft a short insight on the {domain} trend, linking it to a practical next step."
