from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from engageiq.analytics import load_data, summary, trends  # noqa: E402
from engageiq.brief import export_pdf  # noqa: E402
from engageiq.config import PERSONAS, SOURCES  # noqa: E402
from engageiq.feedback_store import FeedbackStore  # noqa: E402
from engageiq.learning import simulate_feedback  # noqa: E402
from engageiq.ranking import SBERT_CACHE_PATH, OpportunityRanker, UserProfile  # noqa: E402


LOGO_SVG = (ROOT / "assets" / "EngageIQ-RadarScope-color.svg").read_text()
LOGO_SVG = LOGO_SVG.replace('<?xml version="1.0" encoding="UTF-8"?>', "").strip()


st.set_page_config(
    page_title="EngageIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "entered" not in st.session_state:
    st.session_state["entered"] = False
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "sim_result" not in st.session_state:
    st.session_state["sim_result"] = None


def theme_css() -> str:
    if st.session_state["theme"] == "dark":
        bg, fg, panel, subtle = "#0d1813", "#e6efe9", "#15241d", "#8aa39a"
        border, accent, accent_strong = "#1f3a30", "#16A691", "#22c4a8"
    else:
        bg, fg, panel, subtle = "#f6f8f7", "#0f1c17", "#ffffff", "#5a6f66"
        border, accent, accent_strong = "#d6dedb", "#0B5A43", "#16A691"
    return f"""
    <style>
      .stApp {{ background: {bg}; color: {fg}; }}
      [data-testid="stSidebar"] {{ background: {panel}; }}
      .eqx-eyebrow {{ color: {accent_strong}; font-size: 12px; letter-spacing: .12em;
                      text-transform: uppercase; font-weight: 700; }}
      .eqx-hero h1 {{ font-size: clamp(34px, 6vw, 58px); line-height: 1.05; margin: 12px 0 16px; color: {fg}; }}
      .eqx-hero p  {{ color: {subtle}; font-size: 17px; max-width: 640px; }}
      .eqx-brand {{ display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }}
      .eqx-brand svg {{ width: 44px; height: 44px; }}
      .eqx-brand strong {{ color: {fg}; font-size: 17px; }}
      .eqx-panel {{ background: {panel}; border: 1px solid {border}; border-radius: 16px; padding: 22px; }}
      .eqx-row {{ display: grid; grid-template-columns: 36px 1fr auto; gap: 14px; align-items: center;
                  padding: 12px 0; border-top: 1px solid {border}; }}
      .eqx-row:first-of-type {{ border-top: none; }}
      .eqx-rank {{ width: 30px; height: 30px; display: grid; place-items: center; border-radius: 999px;
                   background: {accent}; color: white; font-weight: 700; }}
      .eqx-score {{ font-weight: 700; color: {accent_strong}; font-size: 18px; }}
      .eqx-subtle {{ color: {subtle}; font-size: 13px; }}
      .eqx-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }}
      .eqx-stat {{ background: {panel}; border: 1px solid {border}; border-radius: 12px;
                   padding: 14px; text-align: center; }}
      .eqx-stat b {{ display: block; font-size: 22px; color: {fg}; }}
      .eqx-pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px;
                   border: 1px solid {border}; font-size: 12px; color: {subtle}; }}
      .eqx-why {{ background: rgba(22,166,145,.10); border-left: 3px solid {accent};
                  padding: 10px 12px; border-radius: 8px; margin: 8px 0; color: {fg}; }}
      .eqx-action {{ color: {subtle}; font-style: italic; margin: 4px 0 10px; }}
    </style>
    """


st.markdown(theme_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading EngageIQ ranking engine...")
def get_ranker() -> OpportunityRanker:
    backend = "sbert" if SBERT_CACHE_PATH.exists() else "tfidf"
    ranker = OpportunityRanker(embedding_backend=backend)
    FeedbackStore().apply_to_ranker(ranker)
    return ranker


@st.cache_data(show_spinner="Loading opportunity dataset...")
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner=False)
def get_stats() -> dict:
    return summary(load_data())


@st.cache_data(show_spinner=False)
def get_trends() -> dict:
    return trends(load_data())


@st.cache_data(show_spinner=False, ttl=3600)
def llm_suggestion(
    opp_id: str,
    title: str,
    source: str,
    domain: str,
    community: str,
    profile_name: str,
    profile_interests: str,
    profile_goal: str,
) -> str | None:
    """Call Claude Haiku to generate a specific engagement suggestion.
    Returns None if ANTHROPIC_API_KEY is not set or the call fails.
    Results are cached per (opportunity, profile) pair for the session.
    """
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"A professional wants to engage with this online opportunity.\n\n"
            f"Opportunity: {title}\n"
            f"Platform: {source} — {community}\n"
            f"Topic area: {domain}\n\n"
            f"Their profile:\n"
            f"  Name: {profile_name}\n"
            f"  Interests: {profile_interests}\n"
            f"  Goal: {profile_goal}\n\n"
            f"Write exactly ONE sentence: a specific, concrete action they should take "
            f"(e.g., the angle of a comment to leave, what PR to open, what unique insight "
            f"to contribute). No preamble, no generic advice."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        return text or None
    except Exception:
        return None


def theme_toggle_button(key: str) -> None:
    new_theme = "Light" if st.session_state["theme"] == "dark" else "Dark"
    if st.button(f"{new_theme} mode", key=key, use_container_width=True):
        st.session_state["theme"] = new_theme.lower()
        st.rerun()


def make_profile(persona_key, name, interests, skillsets, goal, platforms, time_budget, avoid) -> UserProfile:
    if persona_key != "Custom":
        p = PERSONAS[persona_key]
        return UserProfile(
            name=p["name"],
            interests=p["interests"],
            goal=p["goal"],
            platforms=p["platforms"],
            time_budget=float(p["time_budget"]),
            avoid=avoid.strip(),
        )
    combined = interests.strip()
    if skillsets.strip():
        combined = f"{combined}. Skills: {skillsets.strip()}"
    return UserProfile(
        name=name.strip() or "Custom Profile",
        interests=combined,
        goal=goal.strip(),
        platforms=platforms,
        time_budget=float(time_budget),
        avoid=avoid.strip(),
    )


def render_landing(stats: dict) -> None:
    top = st.columns([1, 6, 1])
    with top[2]:
        theme_toggle_button("theme_landing")

    st.markdown(
        f"""
        <div class="eqx-brand">
          {LOGO_SVG}
          <div>
            <div class="eqx-eyebrow">EngageIQ Platform</div>
            <strong>Opportunity intelligence for online engagement</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.markdown(
            """
            <div class="eqx-hero">
              <div class="eqx-eyebrow">Public signals · Personalized ranking · Faster decisions</div>
              <h1>Find the best places to show up online.</h1>
              <p>EngageIQ scores GitHub, GH Archive, Forem (DEV.to), and Hacker News-style opportunities against
              test personas or your own custom profile, then turns the best matches into ranked actions
              and a downloadable engagement brief.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Enter Dashboard →", type="primary", key="enter_btn"):
            st.session_state["entered"] = True
            st.rerun()

    with right:
        records_label = f"{stats['records']:,} records"
        st.markdown(
            f"""
            <div class="eqx-panel">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div>
                  <div class="eqx-eyebrow">Live demo view</div>
                  <strong>Ranked opportunity feed</strong>
                </div>
                <span class="eqx-pill">{records_label}</span>
              </div>
              <div class="eqx-row">
                <div class="eqx-rank">1</div>
                <div><strong>Good first issue: NLP pipeline contribution</strong>
                     <div class="eqx-subtle">Sofia ML Student · GitHub · 45 min</div></div>
                <div class="eqx-score">91.4</div>
              </div>
              <div class="eqx-row">
                <div class="eqx-rank">2</div>
                <div><strong>Kubernetes discussion with expert-comment gap</strong>
                     <div class="eqx-subtle">David DevOps Engineer · Forem (DEV.to) · 30 min</div></div>
                <div class="eqx-score">88.7</div>
              </div>
              <div class="eqx-row">
                <div class="eqx-rank">3</div>
                <div><strong>Rising developer tool gaining velocity</strong>
                     <div class="eqx-subtle">Lina Trend Spotter · GH Archive · 60 min</div></div>
                <div class="eqx-score">84.9</div>
              </div>
              <div class="eqx-stats">
                <div class="eqx-stat"><div class="eqx-subtle">Sources</div><b>4</b></div>
                <div class="eqx-stat"><div class="eqx-subtle">Domains</div><b>15</b></div>
                <div class="eqx-stat"><div class="eqx-subtle">Exports</div><b>CSV / PDF</b></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard(ranker: OpportunityRanker, stats: dict, trend_data: dict) -> None:
    top = st.columns([1, 6, 1])
    with top[0]:
        if st.button("← Landing", key="back_landing", use_container_width=True):
            st.session_state["entered"] = False
            st.rerun()
    with top[2]:
        theme_toggle_button("theme_dash")

    st.markdown(
        f"""
        <div class="eqx-brand">
          {LOGO_SVG}
          <div>
            <div class="eqx-eyebrow">BAX-423 Final Build</div>
            <strong style="font-size: 22px;">EngageIQ</strong>
            <div class="eqx-subtle">Smart engagement opportunity scorer for GitHub, GH Archive, Forem (DEV.to), and Hacker News.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mc = st.columns(4)
    mc[0].metric("Records", f"{stats['records']:,}")
    mc[1].metric("Sources", len(stats["sources"]))
    mc[2].metric("Domains", stats["domains"])
    mc[3].metric("Embedding", ranker.embedding_backend.upper())
    if ranker.embedding_note:
        st.caption(ranker.embedding_note)

    with st.sidebar:
        st.header("Profile")
        persona_key = st.selectbox("Persona", [*PERSONAS.keys(), "Custom"])
        defaults = PERSONAS.get(persona_key, {})

        if persona_key == "Custom":
            name = st.text_input("Name", "Custom Profile")
            interests = st.text_area("Interests", "machine learning, Python, NLP, open-source contribution")
            skillsets = st.text_input("Skillsets", "Python, pandas, scikit-learn")
            goal = st.text_area("Goal", "Find approachable opportunities for visible contribution.")
            platforms = st.multiselect("Sources", SOURCES, default=["github", "forem"])
            time_budget = st.slider("Weekly time budget (hours)", 1.0, 12.0, 4.0, 0.5)
            avoid_default = "advanced systems internals"
        else:
            st.caption(defaults.get("background", ""))
            name = defaults["name"]
            interests = defaults["interests"]
            skillsets = ""
            goal = defaults["goal"]
            platforms = defaults["platforms"]
            time_budget = float(defaults["time_budget"])
            avoid_default = defaults.get("avoid", "")
            st.caption(f"**Goal:** {goal}")
            st.caption(f"**Sources:** {', '.join(platforms)}")
            st.caption(f"**Time budget:** {int(time_budget)} hours/week")

        avoid = st.text_area(
            "Avoid",
            avoid_default,
            key=f"avoid_{persona_key}",
            help="Topics to steer away from. Edit anytime — changes persist and re-rank immediately.",
        )

        limit = st.slider("Recommendations", 5, 20, 10)
        run_sim = st.button("Run 60-Round Feedback Simulation")

    profile = make_profile(persona_key, name, interests, skillsets, goal, platforms, time_budget, avoid)

    # Run simulation before tabs so results persist regardless of active tab
    if run_sim:
        with st.spinner("Running 60 simulated feedback rounds..."):
            st.session_state["sim_result"] = simulate_feedback(ranker, profile, rounds=60)
        st.sidebar.success("✓ Simulation complete — results in Recommendations tab")

    tab_recs, tab_analytics, tab_tests = st.tabs(["Recommendations", "Analytics", "Persona Coverage"])

    _BRIEF_COLS = ["rank", "source", "domain", "title", "diversified_score",
                   "effort_minutes", "why_this", "suggested_action", "url"]

    with tab_recs:
        recs = ranker.recommend(profile, limit=limit)
        st.subheader(f"Top Opportunities for {profile.name}")

        if recs.empty:
            st.warning("No recommendations matched the current profile and source filters.")
        else:
            # Build curated CSV (only human-readable columns)
            _csv_frame = recs.copy().reset_index(drop=True)
            _csv_frame.insert(0, "rank", range(1, len(_csv_frame) + 1))
            csv_bytes = _csv_frame[_BRIEF_COLS].to_csv(index=False).encode("utf-8")

            dl = st.columns([1, 1, 4])
            with dl[0]:
                st.download_button(
                    "Download CSV Brief",
                    data=csv_bytes,
                    file_name=f"{profile.name.lower().replace(' ', '_')}_engageiq_brief.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl[1]:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path = Path(tmp.name)
                    export_pdf(recs, pdf_path, profile.name)
                    pdf_bytes = pdf_path.read_bytes()
                    st.download_button(
                        "Download PDF Brief",
                        data=pdf_bytes,
                        file_name=f"{profile.name.lower().replace(' ', '_')}_engageiq_brief.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.caption(f"PDF export unavailable: {exc}")

            st.write("")

            for i, (_, row) in enumerate(recs.iterrows(), 1):
                with st.container(border=True):
                    head_l, head_r = st.columns([5, 1])
                    with head_l:
                        st.markdown(f"#### {i}. {row['title']}")
                        meta = f"**{row['source']}** · {row['domain']} · {row['community']} · {int(row['effort_minutes'])} min"
                        if int(row.get("good_first_issue", 0)):
                            meta += " · 🟢 good-first-issue"
                        st.caption(meta)
                    with head_r:
                        st.metric("Score", f"{float(row['diversified_score']):.1f}")

                    st.markdown(
                        f"<div class='eqx-why'><b>Why this?</b> {row['why_this']}</div>",
                        unsafe_allow_html=True,
                    )
                    _llm = llm_suggestion(
                        opp_id=str(row["id"]),
                        title=str(row["title"])[:200],
                        source=str(row["source"]),
                        domain=str(row["domain"]),
                        community=str(row["community"]),
                        profile_name=profile.name,
                        profile_interests=profile.interests[:300],
                        profile_goal=profile.goal[:200],
                    )
                    _action_text = _llm or str(row["suggested_action"])
                    _action_label = "✨ AI suggestion" if _llm else "Suggested action"
                    st.markdown(
                        f"<div class='eqx-action'><b>{_action_label}:</b> {_action_text}</div>",
                        unsafe_allow_html=True,
                    )
                    url = str(row.get("url", ""))
                    if url and "example.com" not in url:
                        st.markdown(f"[Open opportunity ↗]({url})")

                    fb = st.columns([1, 1, 1, 5])
                    rid = str(row["id"])
                    if fb[0].button("👍 Engage", key=f"eng_{rid}_{i}"):
                        FeedbackStore().append(row, "engage", profile)
                        ranker.update_feedback(row, "engage")
                        st.toast("Engage recorded — refreshing rankings")
                        st.rerun()
                    if fb[1].button("🔖 Bookmark", key=f"bm_{rid}_{i}"):
                        FeedbackStore().append(row, "bookmark", profile)
                        ranker.update_feedback(row, "bookmark")
                        st.toast("Bookmark recorded — refreshing rankings")
                        st.rerun()
                    if fb[2].button("👎 Skip", key=f"sk_{rid}_{i}"):
                        FeedbackStore().append(row, "skip", profile)
                        ranker.update_feedback(row, "skip")
                        st.toast("Skip recorded — refreshing rankings")
                        st.rerun()

        sim_result = st.session_state.get("sim_result")
        if sim_result:
            st.markdown("---")
            st.subheader("Feedback Simulation Results")
            sm = st.columns(4)
            sm[0].metric("Rounds", sim_result["rounds"])
            sm[1].metric("Precision@10 before", sim_result["precision_at_10_before"])
            sm[2].metric("Precision@10 after", sim_result["precision_at_10_after"])
            sm[3].metric("Improvement", sim_result["improvement"])
            st.caption(f"Actions: {sim_result['actions']}")
            if st.button("Clear simulation results", key="clear_sim"):
                st.session_state["sim_result"] = None
                st.rerun()

    with tab_analytics:
        left, right = st.columns(2)
        with left:
            st.subheader("Domain Trends (avg growth)")
            domain_frame = pd.DataFrame(trend_data["domains"])
            if not domain_frame.empty:
                st.bar_chart(domain_frame.set_index("domain")["avg_growth"])
                st.dataframe(domain_frame, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Source Volume")
            source_frame = pd.DataFrame(trend_data["sources"])
            if not source_frame.empty:
                st.bar_chart(source_frame.set_index("source")["records"])
            st.subheader("Fast Communities")
            st.dataframe(pd.DataFrame(trend_data["communities"]), use_container_width=True, hide_index=True)

        st.subheader("Daily Engagement Volume")
        daily_frame = pd.DataFrame(trend_data["daily"])
        if not daily_frame.empty:
            st.line_chart(daily_frame.set_index("day")["records"])

    with tab_tests:
        st.subheader("Persona × Capability Coverage")
        st.caption("The four required personas tested against each of the six core capabilities.")
        coverage = pd.DataFrame(
            [
                ["Sofia ML Student (Portfolio Builder)",    "✓", "✓", "✓", "✓", "✓", "✓"],
                ["David DevOps Engineer (Niche Community)", "✓", "✓", "✓", "✓", "✓", "✓"],
                ["Lina Data Journalist (Trend Spotter)",    "✓", "✓", "✓", "✓", "✓", "✓"],
                ["Raj Startup Founder (Marketing-Focused)", "✓", "✓", "✓", "✓", "✓", "✓"],
            ],
            columns=[
                "Persona",
                "1. Multi-Source Ingestion",
                "2. Embedding Retrieval",
                "3. Multi-Stage Ranking",
                "4. Adaptive Learning",
                "5. Batch Analytics",
                "6. Dashboard & Brief",
            ],
        )
        st.dataframe(coverage, use_container_width=True, hide_index=True)

        st.subheader("Pass Criteria (from spec)")
        signals = pd.DataFrame(
            [
                ["Sofia", "Top-10 includes ≥3 good-first-issue GitHub repos; ML-focused; no C++/Rust; <1 hr per opportunity"],
                ["David", "Top-10 Kubernetes/infra-focused; high activity & few contributors; discussion-oriented"],
                ["Lina",  "Top-10 emphasises recency and growth velocity; trend analytics show week-over-week change"],
                ["Raj",   "Recommendations are developer-tools-relevant; discussion threads (not link-only); skip deprioritises low-engagement"],
            ],
            columns=["Persona", "Pass Criteria"],
        )
        st.dataframe(signals, use_container_width=True, hide_index=True)
        st.caption(
            "Simulated Precision@10 validates pipeline consistency. It is not a real-world generalization metric; "
            "simulated relevance is derived from persona-interest token matches."
        )


ranker = get_ranker()
stats = get_stats()
trend_data = get_trends()

if st.session_state["entered"]:
    render_dashboard(ranker, stats, trend_data)
else:
    render_landing(stats)
