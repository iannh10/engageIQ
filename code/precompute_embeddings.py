"""Build the EngageIQ dense embedding cache.

Run this once after installing sentence-transformers. The dashboard will reuse
the cached matrix on later starts.
"""

from __future__ import annotations

from engageiq.ranking import OpportunityRanker


def main() -> None:
    ranker = OpportunityRanker(embedding_backend="sbert")
    print(ranker.embedding_note)
    print(f"Backend: {ranker.embedding_backend}")
    print(f"Records embedded: {len(ranker.df):,}")


if __name__ == "__main__":
    main()
