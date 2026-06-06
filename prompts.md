# Key AI Prompts

_AI tools used: Claude Code (Anthropic) as the primary pair-programming assistant, and Codex by OpenAI as a secondary assistant._

1. Inspect `BAX423_FinalProject_EngageIQ_Spring2026_v2.docx` and `BAX423_Final_OnePager.pdf`; identify required capabilities, rubric criteria, data sources, test personas, and whether the app needs to track personal platform activity.
2. Build a runnable EngageIQ application matching Option C: multi-source ingestion schema, offline 10,000+ record dataset, embedding-style retrieval, engagement ranking, feedback learning, trend analytics, dashboard, and CSV/PDF brief export.
3. Validate the app through local API checks, persona evaluation, and browser interaction testing.
4. Upgrade the retrieval pipeline from TF-IDF fallback retrieval to Sentence-BERT dense embeddings with a cached local embedding matrix and benchmark the four provided personas.
5. Add durable feedback logging so engage/bookmark/skip events are saved to `data/feedback_events.json`, replayed into ranking weights on startup, and documented as part of adaptive learning.
6. Refine the UI into a more complete EngageIQ platform experience with a landing page, light/dark mode, custom profile scoring, saved profiles, logo integration, and a cleaner one-button landing call to action.
7. Design the multi-source ingestion schema and offline snapshot so the dataset clears the 10,000+ record / 15-domain bar, replacing the original Reddit/PRAW source with Forem (DEV.to) alongside GitHub, GH Archive, and Hacker News-style signals.
8. Add Claude Haiku LLM "ideas on how to engage" suggestions: read `ANTHROPIC_API_KEY` from Streamlit secrets, generate a short actionable engagement idea per top opportunity, and fall back gracefully when no key is configured.
9. Implement and run a 60-round simulated feedback loop that replays engage/bookmark/skip signals into the ranking weights and reports the measurable lift in match quality before vs. after learning.
10. Generate the engagement brief as a downloadable PDF (≤4 pages) covering the problem, data pipeline, ranking and adaptive-learning methods, results, and deployment.
11. Deploy the app to Streamlit Community Cloud with a public URL, configure `ANTHROPIC_API_KEY` in cloud secrets, and verify the hosted build serves the brand favicon and renders the dashboard end to end.
