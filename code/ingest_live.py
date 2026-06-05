from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engageiq.collectors import (
    append_records,
    collect_gh_archive_hour,
    collect_github_search,
    collect_hacker_news,
    collect_reddit_praw,
)
from engageiq.config import DOMAINS
from engageiq.data_generator import DATA_PATH, ensure_snapshot


DEFAULT_SUBREDDITS = [
    "MachineLearning",
    "devops",
    "kubernetes",
    "programming",
    "startups",
    "SideProject",
]


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=value pairs from a local .env file if present."""

    if not path.exists():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def default_gh_archive_url(hours_back: int = 2) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    return f"https://data.gharchive.org/{dt:%Y-%m-%d-%H}.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public live EngageIQ opportunities and append them to the offline snapshot."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["github", "gh_archive", "hacker_news"],
        choices=["github", "gh_archive", "hacker_news", "reddit"],
        help="Sources to collect. Reddit requires PRAW credentials.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["Machine Learning", "DevOps/K8s", "Developer Tools", "Python Data Eng"],
        help="Domains to search on GitHub. Use 'all' for every required domain.",
    )
    parser.add_argument("--per-domain", type=int, default=25, help="GitHub records per domain.")
    parser.add_argument("--hn-limit", type=int, default=50, help="Hacker News top-story records to collect.")
    parser.add_argument("--gh-archive-url", default=None, help="Specific GH Archive .json.gz hourly URL.")
    parser.add_argument("--gh-hours-back", type=int, default=2, help="Recent GH Archive hour to use when URL is omitted.")
    parser.add_argument("--gh-lookback-hours", type=int, default=24, help="How many older GH Archive hours to try if recent files are missing.")
    parser.add_argument("--reddit-query", default="developer tools", help="PRAW search query.")
    parser.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS, help="Subreddits for PRAW search.")
    parser.add_argument("--reddit-limit", type=int, default=25, help="Records per subreddit.")
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="CSV dataset path to append to.")
    parser.add_argument("--dry-run", action="store_true", help="Collect and print counts without writing.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    ensure_snapshot(args.output)
    domains = DOMAINS if args.domains == ["all"] else args.domains
    rows: list[dict[str, object]] = []
    source_counts: dict[str, int] = {}

    if "github" in args.sources:
        for domain in domains:
            print(f"Collecting GitHub repositories for {domain}...")
            collected = collect_github_search(domain, limit=args.per_domain)
            rows.extend(collected)
            source_counts["github"] = source_counts.get("github", 0) + len(collected)

    if "gh_archive" in args.sources:
        if args.gh_archive_url:
            print(f"Collecting GH Archive events from {args.gh_archive_url}...")
            collected = collect_gh_archive_hour(args.gh_archive_url)
        else:
            collected = []
            last_error: Exception | None = None
            for offset in range(args.gh_hours_back, args.gh_hours_back + args.gh_lookback_hours):
                url = default_gh_archive_url(offset)
                print(f"Collecting GH Archive events from {url}...")
                try:
                    collected = collect_gh_archive_hour(url)
                    print(f"  Success: collected {len(collected)} GH Archive events from {url}")
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"  Skipped unavailable hour: {exc}")
            if not collected and last_error:
                raise RuntimeError(
                    f"No GH Archive files were available in the last {args.gh_lookback_hours} attempted hours."
                ) from last_error
        rows.extend(collected)
        source_counts["gh_archive"] = source_counts.get("gh_archive", 0) + len(collected)

    if "hacker_news" in args.sources:
        print("Collecting Hacker News top stories...")
        collected = collect_hacker_news(limit=args.hn_limit)
        rows.extend(collected)
        source_counts["hacker_news"] = source_counts.get("hacker_news", 0) + len(collected)

    if "reddit" in args.sources:
        for subreddit in args.subreddits:
            print(f"Collecting Reddit r/{subreddit} results for query: {args.reddit_query!r}...")
            collected = collect_reddit_praw(subreddit, query=args.reddit_query, limit=args.reddit_limit)
            rows.extend(collected)
            source_counts["reddit"] = source_counts.get("reddit", 0) + len(collected)

    print("Collected source counts:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print(f"Total collected before append: {len(rows)}")

    if args.dry_run:
        print("Dry run only; dataset was not changed.")
        return

    total = append_records(rows, args.output)
    print(f"Dataset after append/dedup: {total} rows")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
