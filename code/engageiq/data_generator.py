"""Offline data generation and ingestion helpers for EngageIQ.

The project spec expects live platform ingestion plus an offline snapshot for
grading. This module creates a deterministic 10k+ snapshot that mirrors the
schema returned by GitHub, GH Archive, Forem (DEV.to), and Hacker News collectors.
Live collectors can append into the same CSV later.
"""

from __future__ import annotations

import csv
import hashlib
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from engageiq.config import DOMAIN_KEYWORDS, DOMAINS, SOURCES
else:
    from .config import DOMAIN_KEYWORDS, DOMAINS, SOURCES


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "engageiq_opportunities.csv"


COMMUNITIES = {
    "github": ["issues", "pull-requests", "discussions", "good-first-issues"],
    "gh_archive": ["push-events", "watch-events", "issue-events", "release-events"],
    "forem": ["DEV.to / machinelearning", "DEV.to / devops", "DEV.to / kubernetes", "DEV.to / programming", "DEV.to / startups", "DEV.to / opensource"],
    "hacker_news": ["Ask HN", "Show HN", "Launch HN", "Front Page"],
}

LANGUAGES = ["Python", "TypeScript", "Go", "Rust", "C++", "Java", "Swift", "SQL", "Shell"]


def stable_id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_record(i: int, rng: random.Random) -> dict[str, object]:
    domain = DOMAINS[i % len(DOMAINS)]
    source = SOURCES[i % len(SOURCES)]
    keywords = DOMAIN_KEYWORDS[domain]
    primary = rng.choice(keywords)
    secondary = rng.choice(keywords)
    community = rng.choice(COMMUNITIES[source])
    language = rng.choice(LANGUAGES)

    if domain in {"Machine Learning", "Python Data Eng", "AI Research", "Beginner Coding"}:
        language = rng.choice(["Python", "SQL", "TypeScript"])
    elif domain == "DevOps/K8s":
        language = rng.choice(["Go", "Shell", "Python"])
    elif domain == "GameDev (C++)":
        language = "C++"
    elif domain == "Mobile Dev (iOS/Flutter)":
        language = rng.choice(["Swift", "TypeScript", "Java"])

    created_at = datetime.now(timezone.utc) - timedelta(hours=rng.randint(1, 24 * 35))
    base_popularity = rng.randint(1, 800)
    comments = max(0, int(rng.gauss(base_popularity / 14, 12)))
    stars = max(0, int(rng.gauss(base_popularity * 3, 350))) if source in {"github", "gh_archive"} else 0
    forks = max(0, int(stars * rng.uniform(0.02, 0.22)))
    contributors = max(1, int(rng.gauss(8, 9)))
    open_issues = max(0, int(rng.gauss(35, 30)))
    score = max(1, int(rng.gauss(base_popularity, 180)))
    activity = min(1.0, (comments + open_issues + contributors * 4 + score / 20) / 220)
    growth_rate = round(rng.uniform(0.01, 0.95), 3)
    good_first = domain in {"Machine Learning", "Beginner Coding", "Developer Tools", "Python Data Eng"} and rng.random() < 0.38

    title_templates = [
        "{primary} project needs feedback on {secondary}",
        "Open discussion: practical {primary} patterns for {domain}",
        "Help wanted: improve {primary} workflow in {language}",
        "Rising {domain} tool asks for community contributors",
        "{primary} benchmark thread with actionable follow-ups",
    ]
    title = rng.choice(title_templates).format(
        primary=primary.title(),
        secondary=secondary,
        domain=domain,
        language=language,
    )
    if good_first:
        title = "Good first issue: " + title

    body = (
        f"{domain} opportunity from {community}. The item discusses {primary}, {secondary}, "
        f"and practical next steps for contributors or commenters. Activity is {activity:.2f}; "
        f"growth signal is {growth_rate:.2f}; estimated effort is tuned for focused weekly engagement."
    )
    tags = sorted({domain.lower(), primary, secondary, language.lower(), community.lower().replace("/", "_")})
    url = f"https://example.com/{source}/{stable_id(i, source, domain)}"

    return {
        "id": stable_id(i, source, domain, title),
        "source": source,
        "domain": domain,
        "community": community,
        "title": title,
        "body": body,
        "url": url,
        "author": f"user_{rng.randint(1000, 9999)}",
        "created_at": created_at.isoformat(),
        "score": score,
        "comments": comments,
        "stars": stars,
        "forks": forks,
        "open_issues": open_issues,
        "contributors": contributors,
        "good_first_issue": int(good_first),
        "language": language,
        "effort_minutes": rng.choice([20, 30, 45, 60, 90, 120, 180]),
        "growth_rate": growth_rate,
        "activity": round(activity, 3),
        "toxicity": round(rng.uniform(0.0, 0.35), 3),
        "tags": ",".join(tags),
    }


def generate_snapshot(path: Path = DATA_PATH, records: int = 10500) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(423)
    rows = [make_record(i, rng) for i in range(records)]

    # Deduplicate exactly as the ingestion pipeline would.
    unique = {row["id"]: row for row in rows}
    rows = list(unique.values())
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def ensure_snapshot(path: Path = DATA_PATH, records: int = 10500) -> Path:
    if not path.exists():
        return generate_snapshot(path, records)
    return path


if __name__ == "__main__":
    output = generate_snapshot()
    print(f"Wrote {output}")
