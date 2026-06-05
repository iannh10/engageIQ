"""Optional live collectors for public platform data.

The app runs offline from ``data/engageiq_opportunities.csv`` for grading. These
helpers document and support the live ingestion path required by the project:
GitHub REST, GH Archive, Reddit, and Hacker News can all be normalized into the
same schema as the offline snapshot.
"""

from __future__ import annotations

import gzip
import json
import os
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import DOMAIN_KEYWORDS
from .data_generator import DATA_PATH, stable_id


def http_json(url: str, headers: dict[str, str] | None = None) -> object:
    try:
        import requests

        response = requests.get(url, headers=headers or {"User-Agent": "EngageIQ-BAX423"}, timeout=20)
        response.raise_for_status()
        return response.json()
    except ImportError:
        pass

    req = Request(url, headers=headers or {"User-Agent": "EngageIQ-BAX423"})
    with urlopen(req, timeout=20, context=ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def http_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    try:
        import requests

        response = requests.get(url, headers=headers or {"User-Agent": "EngageIQ-BAX423"}, timeout=30)
        response.raise_for_status()
        return response.content
    except ImportError:
        pass

    req = Request(url, headers=headers or {"User-Agent": "EngageIQ-BAX423"})
    with urlopen(req, timeout=30, context=ssl_context()) as response:
        return response.read()


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def collect_hacker_news(query: str = "developer tools", limit: int = 50) -> list[dict[str, object]]:
    top_ids = http_json("https://hacker-news.firebaseio.com/v0/topstories.json")
    rows = []
    for item_id in list(top_ids)[:limit]:
        item = http_json(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json")
        title = item.get("title", "")
        text = f"{title} {query}"
        rows.append(normalize_record("hacker_news", title, text, "Hacker News", item.get("url", ""), item.get("score", 0), item.get("descendants", 0)))
    return rows


def collect_github_search(domain: str, limit: int = 50) -> list[dict[str, object]]:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"User-Agent": "EngageIQ-BAX423"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    keywords = " ".join(DOMAIN_KEYWORDS[domain][:3])
    params = urlencode({"q": f"{keywords} good-first-issues", "sort": "updated", "per_page": min(limit, 100)})
    data = http_json(f"https://api.github.com/search/repositories?{params}", headers=headers)
    rows = []
    for item in data.get("items", []):
        body = item.get("description") or ""
        rows.append(
            normalize_record(
                "github",
                item.get("full_name", ""),
                body,
                domain,
                item.get("html_url", ""),
                item.get("stargazers_count", 0),
                item.get("open_issues_count", 0),
                stars=item.get("stargazers_count", 0),
                language=item.get("language") or "",
            )
        )
    return rows


def collect_gh_archive_hour(url: str) -> list[dict[str, object]]:
    rows = []
    raw = gzip.decompress(http_bytes(url, headers={"User-Agent": "EngageIQ-BAX423"})).decode("utf-8")
    for line in raw.splitlines():
        event = json.loads(line)
        repo = event.get("repo", {}).get("name", "")
        event_type = event.get("type", "")
        rows.append(normalize_record("gh_archive", repo, event_type, "GH Archive", f"https://github.com/{repo}", 1, 0))
    return rows


def collect_reddit_praw(subreddit: str, query: str = "developer tools", limit: int = 50) -> list[dict[str, object]]:
    """Collect Reddit submissions through PRAW.

    Required environment variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    - REDDIT_USER_AGENT

    The function intentionally reads public subreddit/search results only. It
    does not access private account history, saved posts, votes, or messages.
    """

    try:
        import praw
    except ImportError as exc:
        raise RuntimeError("PRAW is not installed. Run `pip install praw` or add it to your environment.") from exc

    missing = [
        name
        for name in ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"Missing Reddit environment variables: {', '.join(missing)}")

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    rows = []
    for submission in reddit.subreddit(subreddit).search(query, sort="new", limit=limit):
        body = f"{submission.selftext or ''} subreddit:{subreddit}"
        rows.append(
            normalize_record(
                "reddit",
                submission.title,
                body,
                f"r/{subreddit}",
                f"https://reddit.com{submission.permalink}",
                int(getattr(submission, "score", 0) or 0),
                int(getattr(submission, "num_comments", 0) or 0),
            )
        )
    return rows


def normalize_record(
    source: str,
    title: str,
    body: str,
    community: str,
    url: str,
    score: int,
    comments: int,
    stars: int = 0,
    language: str = "",
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    text = f"{title} {body}".lower()
    domain = infer_domain(text)
    return {
        "id": stable_id(source, title, url),
        "source": source,
        "domain": domain,
        "community": community,
        "title": title,
        "body": body,
        "url": url,
        "author": "",
        "created_at": now,
        "score": score,
        "comments": comments,
        "stars": stars,
        "forks": 0,
        "open_issues": comments,
        "contributors": 1,
        "good_first_issue": int("good first" in text or "beginner" in text),
        "language": language,
        "effort_minutes": 60,
        "growth_rate": 0.25,
        "activity": min(1.0, (score + comments) / 500),
        "toxicity": 0.05,
        "tags": f"{domain.lower()},{source},{language.lower()}",
    }


def infer_domain(text: str) -> str:
    best = ("Developer Tools", 0)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best[1]:
            best = (domain, hits)
    return best[0]


def append_records(rows: list[dict[str, object]], path: Path = DATA_PATH) -> int:
    import pandas as pd

    if not rows:
        return 0
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    incoming = pd.DataFrame(rows)
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"])
    merged.to_csv(path, index=False)
    return len(merged)
