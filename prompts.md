# Key AI Prompts

1. Inspect `BAX423_FinalProject_EngageIQ_Spring2026_v2.docx` and `BAX423_Final_OnePager.pdf`; identify required capabilities, rubric criteria, data sources, test personas, and whether the app needs to track personal platform activity.
2. Build a runnable EngageIQ application matching Option C: multi-source ingestion schema, offline 10,000+ record dataset, embedding-style retrieval, engagement ranking, feedback learning, trend analytics, dashboard, and CSV/PDF brief export.
3. Validate the app through local API checks, persona evaluation, and browser interaction testing.
4. Upgrade the retrieval pipeline from TF-IDF fallback retrieval to Sentence-BERT dense embeddings with a cached local embedding matrix and benchmark the four provided personas.
5. Add durable feedback logging so engage/bookmark/skip events are saved to `data/feedback_events.json`, replayed into ranking weights on startup, and documented as part of adaptive learning.
6. Refine the UI into a more complete EngageIQ platform experience with a landing page, light/dark mode, custom profile scoring, saved profiles, logo integration, and a cleaner one-button landing call to action.
