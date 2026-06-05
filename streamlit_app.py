from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from engageiq.analytics import load_data, summary, trends
from engageiq.config import PERSONAS, SOURCES
from engageiq.feedback_store import FeedbackStore
from engageiq.learning import simulate_feedback
from engageiq.ranking import SBERT_CACHE_PATH, OpportunityRanker, UserProfile


st.set_page_config(
    page_title="EngageIQ",
    page_icon="assets/EngageIQ-RadarScope-color.svg",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading EngageIQ ranking engine...")
def get_ranker() -> OpportunityRanker:
    backend = "sbert" if SBERT_CACHE_PATH.exists() else "tfidf"
    ranker = OpportunityRanker(embedding_backend=backend)
    FeedbackStore().apply_to_ranker(ranker)
    return ranker


@st.cache_data(show_spinner="Loading opportunity dataset...")
def get_data() -> pd.DataFrame:
    return load_data()


def make_profile(
    persona_key: str,
    name: str,
    interests: str,
    goal: str,
    platforms: list[str],
    time_budget: float,
    avoid: str,
) -> UserProfile:
    if persona_key != "Custom":
        persona = PERSONAS[persona_key]
        return UserProfile(
            name=persona["name"],
            interests=persona["interests"],
            goal=persona["goal"],
            platforms=persona["platforms"],
            time_budget=float(persona["time_budget"]),
            avoid=persona.get("avoid", ""),
        )
    return UserProfile(
        name=name.strip() or "Custom Profile",
        interests=interests.strip(),
        goal=goal.strip(),
        platforms=platforms,
        time_budget=float(time_budget),
        avoid=avoid.strip(),
    )


def recommendations_table(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source",
        "domain",
        "community",
        "title",
        "diversified_score",
        "effort_minutes",
        "growth_rate",
        "why_this",
        "suggested_action",
        "url",
    ]
    return frame[cols].rename(
        columns={
            "source": "Source",
            "domain": "Domain",
            "community": "Community",
            "title": "Title",
            "diversified_score": "Score",
            "effort_minutes": "Effort Min",
            "growth_rate": "Growth",
            "why_this": "Why This",
            "suggested_action": "Suggested Action",
            "url": "URL",
        }
    )


ranker = get_ranker()
data = get_data()
stats = summary(data)
trend_data = trends(data)

st.title("EngageIQ")
st.caption("Smart engagement opportunity scoring across GitHub, GH Archive, Reddit-style communities, and Hacker News.")

metric_cols = st.columns(4)
metric_cols[0].metric("Records", f"{stats['records']:,}")
metric_cols[1].metric("Sources", len(stats["sources"]))
metric_cols[2].metric("Domains", stats["domains"])
metric_cols[3].metric("Embedding", ranker.embedding_backend.upper())
st.info(ranker.embedding_note)

with st.sidebar:
    st.header("Profile")
    persona_key = st.selectbox("Persona", [*PERSONAS.keys(), "Custom"])
    defaults = PERSONAS.get(persona_key, {})

    if persona_key == "Custom":
        name = st.text_input("Name", "Custom Profile")
        interests = st.text_area("Interests", "beginner Python, web development, documentation, portfolio projects")
        goal = st.text_area("Goal", "Find approachable opportunities for visible contribution.")
        platforms = st.multiselect("Sources", SOURCES, default=["github", "reddit"])
        time_budget = st.slider("Weekly time budget", 1.0, 12.0, 4.0, 0.5)
        avoid = st.text_area("Avoid", "advanced systems internals")
    else:
        st.write(defaults.get("background", ""))
        name = defaults["name"]
        interests = defaults["interests"]
        goal = defaults["goal"]
        platforms = defaults["platforms"]
        time_budget = float(defaults["time_budget"])
        avoid = defaults.get("avoid", "")
        st.caption(f"Goal: {goal}")
        st.caption(f"Sources: {', '.join(platforms)}")

    limit = st.slider("Recommendations", 5, 20, 10)
    run_simulation = st.button("Run 60-Round Feedback Simulation")

profile = make_profile(persona_key, name, interests, goal, platforms, time_budget, avoid)

tab_recs, tab_analytics, tab_tests = st.tabs(["Recommendations", "Analytics", "Persona Coverage"])

with tab_recs:
    recs = ranker.recommend(profile, limit=limit)
    st.subheader(f"Top Opportunities for {profile.name}")
    if recs.empty:
        st.warning("No recommendations matched the current profile and source filters.")
    else:
        st.dataframe(recommendations_table(recs), use_container_width=True, hide_index=True)
        csv = recommendations_table(recs).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Brief",
            data=csv,
            file_name=f"{profile.name.lower().replace(' ', '_')}_engageiq_brief.csv",
            mime="text/csv",
        )

    st.subheader("Feedback")
    if recs.empty:
        st.caption("Run recommendations before recording feedback.")
    else:
        selected_id = st.selectbox(
            "Opportunity",
            recs["id"].tolist(),
            format_func=lambda row_id: str(recs.loc[recs["id"] == row_id, "title"].iloc[0])[:100],
        )
        action = st.radio("Action", ["engage", "bookmark", "skip"], horizontal=True)
        if st.button("Save Feedback"):
            row = ranker.df[ranker.df["id"] == selected_id].iloc[0]
            FeedbackStore().append(row, action, profile)
            ranker.update_feedback(row, action)
            st.success(f"Recorded {action} feedback.")

    if run_simulation:
        result = simulate_feedback(ranker, profile, rounds=60)
        st.write(result)

with tab_analytics:
    left, right = st.columns(2)
    with left:
        st.subheader("Domain Trends")
        domain_frame = pd.DataFrame(trend_data["domains"])
        st.bar_chart(domain_frame.set_index("domain")["avg_growth"])
        st.dataframe(domain_frame, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Source Volume")
        source_frame = pd.DataFrame(trend_data["sources"])
        st.bar_chart(source_frame.set_index("source")["records"])
        st.subheader("Fast Communities")
        st.dataframe(pd.DataFrame(trend_data["communities"]), use_container_width=True, hide_index=True)

with tab_tests:
    st.subheader("Persona Coverage")
    coverage = pd.DataFrame(
        [
            ["Sofia", "ML student", "Machine Learning, Beginner Coding, Python, good-first-issue"],
            ["Emma", "Career switcher", "Beginner Coding, Python/web learning, docs, first contribution"],
            ["David", "DevOps engineer", "DevOps/K8s, Cloud APIs, infrastructure discussions"],
            ["Lina", "Data journalist", "Trending Open-Source, freshness, growth velocity"],
            ["Raj", "Startup founder", "Developer Tools, APIs, CLI tooling, startup-relevant threads"],
        ],
        columns=["Persona", "Use Case", "Expected Ranking Signals"],
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    st.caption(
        "The simulated Precision@10 benchmark validates pipeline consistency. "
        "It is not a real-world generalization metric because simulated relevance is derived from persona-interest matches."
    )
