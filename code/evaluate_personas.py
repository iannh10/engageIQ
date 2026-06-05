from __future__ import annotations

from engageiq.config import PERSONAS
from engageiq.learning import simulate_feedback
from engageiq.ranking import OpportunityRanker, UserProfile


def main() -> None:
    ranker = OpportunityRanker()
    print("persona,top_10_sources,top_10_domains,precision_before,precision_after,improvement")
    for key, persona in PERSONAS.items():
        profile = UserProfile(
            name=persona["name"],
            interests=persona["interests"],
            goal=persona["goal"],
            platforms=persona["platforms"],
            time_budget=persona["time_budget"],
            avoid=persona.get("avoid", ""),
        )
        recs = ranker.recommend(profile, limit=10)
        result = simulate_feedback(ranker, profile, rounds=60)
        sources = "|".join(sorted(recs["source"].unique()))
        domains = "|".join(sorted(recs["domain"].unique()))
        print(
            f"{key},{sources},{domains},{result['precision_at_10_before']},"
            f"{result['precision_at_10_after']},{result['improvement']}"
        )


if __name__ == "__main__":
    main()

