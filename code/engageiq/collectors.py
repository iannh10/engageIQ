"""Optional live collectors for public platform data.

The app runs offline from ``data/engageiq_opportunities.csv`` for grading. These
helpers document and support the live ingestion path required by the project:
GitHub REST, GH Archive, Forem (DEV.to), and Hacker News can all be normalized
into the same schema as the offline snapshot.
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


# Map common Forem (DEV.to) tags onto the 15 required EngageIQ domains so live
# articles get an accurate domain label from their own metadata rather than the
# permissive keyword guesser in ``infer_domain``.
FOREM_TAG_DOMAIN = {
    "machinelearning": "Machine Learning",
    "datascience": "Machine Learning",
    "ml": "Machine Learning",
    "ai": "AI Research",
    "llm": "AI Research",
    "devops": "DevOps/K8s",
    "kubernetes": "DevOps/K8s",
    "k8s": "DevOps/K8s",
    "opensource": "Trending Open-Source",
    "devtools": "Developer Tools",
    "tooling": "Developer Tools",
    "security": "Cybersecurity",
    "cybersecurity": "Cybersecurity",
    "react": "Frontend (React/Web)",
    "webdev": "Frontend (React/Web)",
    "javascript": "Frontend (React/Web)",
    "typescript": "Frontend (React/Web)",
    "saas": "B2B SaaS",
    "startups": "B2B SaaS",
    "blockchain": "Blockchain",
    "web3": "Blockchain",
    "python": "Python Data Eng",
    "dataengineering": "Python Data Eng",
    "gamedev": "GameDev (C++)",
    "cpp": "GameDev (C++)",
    "embedded": "Embedded Systems (C/RTOS)",
    "iot": "Embedded Systems (C/RTOS)",
    "rtos": "Embedded Systems (C/RTOS)",
    "aws": "Cloud APIs",
    "cloud": "Cloud APIs",
    "serverless": "Cloud APIs",
    "azure": "Cloud APIs",
    "flutter": "Mobile Dev (iOS/Flutter)",
    "ios": "Mobile Dev (iOS/Flutter)",
    "android": "Mobile Dev (iOS/Flutter)",
    "beginners": "Beginner Coding",
    "tutorial": "Beginner Coding",
    "codenewbie": "Beginner Coding",
}


def forem_domain(tags: list[str]) -> str | None:
    for tag in tags:
        key = tag.lower().replace(" ", "").replace("-", "").replace("_", "")
        if key in FOREM_TAG_DOMAIN:
            return FOREM_TAG_DOMAIN[key]
    return None


def collect_forem(tag: str, query: str = "", limit: int = 50) -> list[dict[str, object]]:
    """Collect published articles from the Forem (DEV.to) public API.

    No authentication is required to read public articles. The function reads
    published articles for a tag only — it does not access private user data,
    drafts, reading history, or account analytics.

    Endpoint: https://dev.to/api/articles?tag={tag}&per_page={limit}
    """

    params = urlencode({"tag": tag, "per_page": min(limit, 100), "top": 365})
    data = http_json(f"https://dev.to/api/articles?{params}")
    rows = []
    for item in data if isinstance(data, list) else []:
        title = item.get("title", "")
        description = item.get("description") or ""
        raw_tags = item.get("tag_list") or item.get("tags") or []
        if isinstance(raw_tags, str):
            tag_list = [part.strip() for part in raw_tags.split(",") if part.strip()]
        else:
            tag_list = list(raw_tags)
        body = f"{description} tags:{' '.join(tag_list)}"
        if query:
            body = f"{body} {query}"
        rows.append(
            normalize_record(
                "forem",
                title,
                body,
                f"DEV.to / {tag}",
                item.get("url", ""),
                int(item.get("positive_reactions_count", 0) or 0),
                int(item.get("comments_count", 0) or 0),
                domain=forem_domain(tag_list),
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
    domain: str | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    text = f"{title} {body}".lower()
    if domain is None:
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
