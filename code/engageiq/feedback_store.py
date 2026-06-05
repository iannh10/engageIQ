"""Durable feedback event logging for EngageIQ."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_generator import DATA_PATH
from .ranking import OpportunityRanker, UserProfile


FEEDBACK_PATH = DATA_PATH.parent / "feedback_events.json"
VALID_ACTIONS = {"engage", "bookmark", "skip"}


class FeedbackStore:
    def __init__(self, path: Path = FEEDBACK_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("[]")

    def load_events(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def append(self, row: pd.Series, action: str, profile: UserProfile) -> dict[str, Any]:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported feedback action: {action}")
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "opportunity": {
                "id": str(row.get("id", "")),
                "source": str(row.get("source", "")),
                "domain": str(row.get("domain", "")),
                "community": str(row.get("community", "")),
                "title": str(row.get("title", "")),
                "tags": str(row.get("tags", "")),
                "url": str(row.get("url", "")),
            },
            "profile": {
                "name": profile.name,
                "goal": profile.goal,
                "interests": profile.interests,
                "platforms": profile.platforms,
            },
        }
        with self._lock:
            events = self.load_events()
            events.append(event)
            self.path.write_text(json.dumps(events, indent=2))
        return event

    def apply_to_ranker(self, ranker: OpportunityRanker) -> int:
        applied = 0
        events = self.load_events()
        if not events:
            return applied
        rows_by_id = ranker.df.set_index("id", drop=False)
        for event in events:
            action = str(event.get("action", ""))
            opportunity = event.get("opportunity") or {}
            if action not in VALID_ACTIONS or not isinstance(opportunity, dict):
                continue
            row_id = str(opportunity.get("id", ""))
            if row_id not in rows_by_id.index:
                continue
            ranker.update_feedback(rows_by_id.loc[row_id], action)
            applied += 1
        return applied

    def summary(self) -> dict[str, Any]:
        events = self.load_events()
        counts = {action: 0 for action in sorted(VALID_ACTIONS)}
        for event in events:
            action = str(event.get("action", ""))
            if action in counts:
                counts[action] += 1
        return {
            "total_events": len(events),
            "counts": counts,
            "path": str(self.path),
        }
