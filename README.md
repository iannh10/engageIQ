# EngageIQ - BAX 423 Final Project

EngageIQ is a smart engagement opportunity scorer for GitHub, GH Archive, Reddit, and Hacker News. It implements the Option C requirements from `BAX423_Final_OnePager.pdf` and `BAX423_FinalProject_EngageIQ_Spring2026_v2.docx`.

## What It Does

- Ingests and stores a structured offline engagement dataset across all 15 required technical domains.
- Deduplicates records by stable opportunity id.
- Represents opportunities and user profiles with Sentence-BERT dense embeddings when available, with TF-IDF cosine retrieval as a reliable fallback.
- Scores and re-ranks opportunities using relevance, community health, visibility, freshness, effort fit, and feedback affinity.
- Learns from durable `engage`, `skip`, and `bookmark` feedback saved to `data/feedback_events.json`.
- Runs a 60-round simulated feedback benchmark.
- Displays trend analytics by domain, source, community, and time.
- Exports a weekly engagement brief as CSV or PDF.
- Provides a modern landing page, light/dark mode, custom profile input, and saved custom profiles in the browser.

## Run Locally

```bash
python3 code/app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Host On Streamlit Community Cloud

The Streamlit entrypoint is:

```text
streamlit_app.py
```

To test it locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy publicly:

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Select the GitHub repository and branch.
4. Set the app entrypoint file to `streamlit_app.py`.
5. Keep `requirements.txt` at the repository root so Streamlit installs the app dependencies.

The dense SBERT cache file `data/engageiq_sbert_embeddings.npy` is about 113 MB, which is above GitHub's normal single-file limit. For a public Streamlit deployment, either track that file with Git LFS or omit it. The Streamlit wrapper uses SBERT when the cache is present and TF-IDF fallback when it is not, so the hosted app can still launch reliably. Git LFS gives the best demo startup experience.

The first run creates:

```text
data/engageiq_opportunities.csv
```

with 10,500 deterministic offline records if the CSV is missing. The current local dataset may be larger after live ingestion. The app does not require personal platform account tracking.

## Evaluate Personas

```bash
python3 code/evaluate_personas.py
```

This prints a compact pass/fail-support table for the four required personas and shows the adaptive-learning precision change after 60 simulated feedback rounds.

## Dense Embeddings

EngageIQ now supports Sentence-BERT dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the dense embedding cache:

```bash
python3 code/precompute_embeddings.py
```

The first run downloads the SBERT model if it is not already cached and writes:

```text
data/engageiq_sbert_embeddings.npy
data/engageiq_sbert_embeddings.json
```

After that, `python3 code/app.py` reuses the cached dense vectors. If the model is unavailable in a grading environment, the app falls back to TF-IDF retrieval so the dashboard remains runnable.

## Test The Dataset

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("data/engageiq_opportunities.csv", low_memory=False)
print("Rows:", len(df))
print("Unique IDs:", df.id.nunique())
print("Domains:", df.domain.nunique())
print(df.source.value_counts())
PY
```

The EngageIQ requirement is 10,000+ structured records across all 15 technical domains. The current local dataset has 76,972 unique records after live ingestion tests.

## Live API Ingestion

The app runs reliably from the offline snapshot, but you can append live public records with:

```bash
python3 code/ingest_live.py --sources github gh_archive hacker_news
```

GitHub can use unauthenticated search, but a token improves rate limits:

```bash
export GITHUB_TOKEN="your_github_token"
```

You can also copy `.env.example` to `.env` and place credentials there:

```bash
cp .env.example .env
```

Then edit `.env` and set `GITHUB_TOKEN`. The script loads `.env` automatically.

Reddit requires PRAW credentials:

```bash
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="EngageIQ:BAX423:v1.0 by your_username"
python3 code/ingest_live.py --sources reddit --reddit-query "machine learning" --subreddits MachineLearning learnpython
```

Useful examples:

```bash
# Collect GitHub across every required domain
python3 code/ingest_live.py --sources github --domains all --per-domain 20

# Collect one recent GH Archive hour and Hacker News top stories
python3 code/ingest_live.py --sources gh_archive hacker_news --hn-limit 100

# Preview without writing to the CSV
python3 code/ingest_live.py --sources hacker_news --dry-run
```

## Data Sources

The grading app runs from the offline snapshot. `code/ingest_live.py` and `code/engageiq/collectors.py` contain live collection helpers for public data from:

- GitHub REST API v3
- GH Archive
- Hacker News API
- Reddit via PRAW

For privacy, the app is designed to collect public opportunities, not a user's personal GitHub, Reddit, or Hacker News activity.

Current source/API status:

| Source | Offline snapshot | Live ingestion |
|---|---:|---:|
| GitHub | Yes | Yes |
| GH Archive | Yes | Yes |
| Hacker News | Yes | Yes |
| Reddit | Reddit-style snapshot | PRAW script support; requires Reddit approval/credentials |

## Dashboard Behavior

Ranking is deterministic. If the same profile, filters, dataset, and feedback state are used, refreshing the ranking should usually produce the same order. Rankings change when you switch personas, edit a custom profile, change platform filters, give engage/bookmark/skip feedback, or ingest new live data.

Custom profiles can be saved in the browser. Saved profiles appear in the persona dropdown under "Saved custom profiles" and are stored in `localStorage`.

In-app feedback is durable. The dashboard writes engage/bookmark/skip events to `data/feedback_events.json` and replays those events into the ranking weights on startup.

## Project Structure

```text
code/
  app.py                       # Web dashboard and API
  evaluate_personas.py          # Persona evaluation runner
  precompute_embeddings.py      # Optional SBERT dense embedding cache builder
  engageiq/
    analytics.py                # Batch/trend analytics
    brief.py                    # CSV/PDF export
    collectors.py               # Optional live public-data collectors
    config.py                   # Domains and test personas
    data_generator.py           # Offline 10k+ dataset generator
    feedback_store.py           # Durable feedback event log and replay
    learning.py                 # Feedback simulation
    ranking.py                  # Retrieval, scoring, explanations
data/
  engageiq_opportunities.csv    # Generated offline snapshot
  engageiq_sbert_embeddings.npy # Cached SBERT dense embedding matrix
  feedback_events.json          # Durable in-app feedback log, created on first run
docs/
  prompts.md                    # Key AI prompts log
```

## BAX-423 Techniques Used

- Embedding/retrieval: Sentence-BERT dense vector representation and cosine similarity retrieve semantically related opportunities; TF-IDF remains as an offline fallback.
- Multi-stage ranking: candidate generation, composite engagement scoring, and diversity-aware re-ranking.
- Adaptive learning: durable feedback events update recommendation affinity over repeated rounds and across app restarts.
- Batch analytics: aggregate trend detection over the full offline dataset.
