from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from engageiq.analytics import load_data, summary, trends
from engageiq.brief import export_csv, export_pdf
from engageiq.config import PERSONAS, SOURCES
from engageiq.data_generator import ensure_snapshot
from engageiq.feedback_store import FeedbackStore
from engageiq.learning import simulate_feedback
from engageiq.ranking import OpportunityRanker, UserProfile


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "exports"
ASSET_DIR = ROOT / "assets"
HOST = "127.0.0.1"
PORT = 8000

ensure_snapshot()
RANKER = OpportunityRanker()
FEEDBACK_STORE = FeedbackStore()
PERSISTED_FEEDBACK_APPLIED = FEEDBACK_STORE.apply_to_ranker(RANKER)
DATAFRAME = load_data()
LAST_RECOMMENDATIONS: dict[str, pd.DataFrame] = {}


def make_profile(payload: dict[str, object]) -> UserProfile:
    persona_key = str(payload.get("persona", "Sofia"))
    if persona_key == "custom":
        base = {
            "name": "Custom Profile",
            "interests": "",
            "goal": "",
            "platforms": SOURCES,
            "time_budget": 4,
            "avoid": "",
        }
    else:
        base = PERSONAS.get(persona_key, PERSONAS["Sofia"])
    platforms = payload.get("platforms") or base["platforms"]
    if isinstance(platforms, str):
        platforms = [platforms]
    interests = str(payload.get("interests") or base["interests"])
    skillsets = str(payload.get("skillsets") or "").strip()
    if skillsets:
        interests = f"{interests}. Skills: {skillsets}"
    return UserProfile(
        name=str(payload.get("name") or base["name"]),
        interests=interests,
        goal=str(payload.get("goal") or base["goal"]),
        platforms=[p for p in platforms if p in SOURCES],
        time_budget=float(payload.get("time_budget") or base["time_budget"]),
        avoid=str(payload.get("avoid") or base.get("avoid", "")),
    )


class EngageIQHandler(BaseHTTPRequestHandler):
    server_version = "EngageIQ/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
        elif parsed.path == "/api/bootstrap":
            self.send_json({
                "personas": PERSONAS,
                "sources": SOURCES,
                "summary": summary(DATAFRAME),
                "trends": trends(DATAFRAME),
                "embedding": {
                    "backend": RANKER.embedding_backend,
                    "note": RANKER.embedding_note,
                },
                "feedback": {
                    **FEEDBACK_STORE.summary(),
                    "applied_events": PERSISTED_FEEDBACK_APPLIED,
                },
            })
        elif parsed.path.startswith("/assets/"):
            self.serve_asset(parsed.path)
        elif parsed.path.startswith("/download/"):
            self.serve_file(parsed.path)
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        if parsed.path == "/api/recommend":
            profile = make_profile(payload)
            recs = RANKER.recommend(profile, limit=int(payload.get("limit", 10)))
            LAST_RECOMMENDATIONS[profile.name] = recs
            self.send_json({"profile": profile.__dict__, "recommendations": frame_to_json(recs)})
        elif parsed.path == "/api/feedback":
            row_id = str(payload.get("id", ""))
            action = str(payload.get("action", "bookmark"))
            if action not in {"engage", "bookmark", "skip"}:
                self.send_json({"ok": False, "message": "Unsupported feedback action"}, status=400)
                return
            row = RANKER.df[RANKER.df["id"] == row_id]
            if row.empty:
                self.send_json({"ok": False, "message": "Unknown opportunity id"}, status=404)
                return
            profile = make_profile(payload)
            selected = row.iloc[0]
            event = FEEDBACK_STORE.append(selected, action, profile)
            RANKER.update_feedback(selected, action)
            self.send_json({
                "ok": True,
                "message": f"Recorded {action} feedback",
                "event": event,
                "feedback": FEEDBACK_STORE.summary(),
            })
        elif parsed.path == "/api/simulate-learning":
            profile = make_profile(payload)
            result = simulate_feedback(RANKER, profile, rounds=int(payload.get("rounds", 60)))
            self.send_json(result)
        elif parsed.path == "/api/export":
            profile = make_profile(payload)
            recs = LAST_RECOMMENDATIONS.get(profile.name)
            if recs is None or recs.empty:
                recs = RANKER.recommend(profile, limit=10)
            export_type = str(payload.get("type", "csv")).lower()
            safe_name = profile.name.lower().replace(" ", "_").replace("/", "_")
            if export_type == "pdf":
                path = export_pdf(recs, EXPORT_DIR / f"{safe_name}_brief.pdf", profile.name)
            else:
                path = export_csv(recs, EXPORT_DIR / f"{safe_name}_brief.csv")
            self.send_json({"url": f"/download/{path.name}", "path": str(path)})
        else:
            self.send_error(404, "Not found")

    def serve_file(self, request_path: str) -> None:
        name = Path(request_path).name
        path = EXPORT_DIR / name
        if not path.exists():
            self.send_error(404, "Export not found")
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(path.read_bytes())

    def serve_asset(self, request_path: str) -> None:
        name = Path(request_path).name
        path = ASSET_DIR / name
        if not path.exists() or not path.is_file():
            self.send_error(404, "Asset not found")
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(path.read_bytes())

    def send_json(self, data: object, status: int = 200) -> None:
        raw = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def frame_to_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    output = frame.copy()
    cols = [
        "id",
        "source",
        "domain",
        "community",
        "title",
        "url",
        "created_at",
        "score",
        "comments",
        "stars",
        "good_first_issue",
        "language",
        "effort_minutes",
        "growth_rate",
        "activity",
        "health",
        "visibility",
        "freshness",
        "diversified_score",
        "why_this",
        "suggested_action",
    ]
    return output[cols].astype(object).where(pd.notna(output[cols]), None).to_dict(orient="records")


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EngageIQ</title>
  <script>
    const savedTheme = localStorage.getItem('engageiq-theme') || 'light';
    document.documentElement.dataset.theme = savedTheme;
  </script>
  <style>
    :root {
      color-scheme: light;
      --ink: #16201e;
      --muted: #64736f;
      --quiet: #82908c;
      --line: #dbe4e1;
      --soft-line: #edf1ef;
      --paper: #f4f7f6;
      --panel: #ffffff;
      --panel-2: #f9fbfa;
      --field: #ffffff;
      --green: #1f5c51;
      --green-soft: #e7f3ef;
      --blue: #315f99;
      --blue-soft: #edf3fb;
      --coral: #a95042;
      --coral-soft: #fbefec;
      --gold: #8b6e25;
      --gold-soft: #fbf4e3;
      --shadow: 0 1px 2px rgba(18, 28, 26, .05), 0 16px 40px rgba(18, 28, 26, .07);
      --focus: rgba(31, 92, 81, .18);
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --ink: #edf5f2;
      --muted: #a8bbb5;
      --quiet: #7f918c;
      --line: #273c37;
      --soft-line: #1f302c;
      --paper: #0e1715;
      --panel: #14211e;
      --panel-2: #182a26;
      --field: #0f1b18;
      --green: #71d6bd;
      --green-soft: #173a34;
      --blue: #8fb9ee;
      --blue-soft: #172842;
      --coral: #ee9b87;
      --coral-soft: #3a201d;
      --gold: #e7c86c;
      --gold-soft: #332a16;
      --shadow: 0 1px 2px rgba(0, 0, 0, .35), 0 20px 44px rgba(0, 0, 0, .26);
      --focus: rgba(113, 214, 189, .22);
    }
    * { box-sizing: border-box; }
    body {
      --cursor-x: 50%;
      --cursor-y: 42%;
      margin: 0;
      background:
        radial-gradient(circle at 10% -10%, rgba(49, 95, 153, .12), transparent 30%),
        linear-gradient(180deg, var(--paper), var(--paper));
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .landing {
      position: relative;
      min-height: 100vh;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      background:
        linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--green-soft) 68%, transparent)),
        var(--paper);
    }
    .landing::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        radial-gradient(520px circle at var(--cursor-x) var(--cursor-y), color-mix(in srgb, var(--green) 18%, transparent), transparent 58%),
        linear-gradient(90deg, color-mix(in srgb, var(--line) 42%, transparent) 1px, transparent 1px),
        linear-gradient(0deg, color-mix(in srgb, var(--line) 42%, transparent) 1px, transparent 1px);
      background-size: auto, 42px 42px, 42px 42px;
      opacity: .72;
    }
    .landing-nav {
      position: relative;
      z-index: 1;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 20px 28px;
    }
    .landing-main {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
      align-items: center;
      gap: 42px;
      width: min(1180px, calc(100% - 48px));
      margin: 0 auto;
      padding: 42px 0 64px;
    }
    .hero-copy h1 {
      max-width: 760px;
      margin: 10px 0 14px;
      font-size: clamp(38px, 6vw, 76px);
      line-height: .98;
      letter-spacing: 0;
    }
    .hero-copy p {
      max-width: 640px;
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.6;
    }
    .hero-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 26px;
    }
    .hero-actions button {
      min-height: 44px;
      padding: 12px 18px;
    }
    .platform-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: color-mix(in srgb, var(--panel) 86%, transparent);
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(16px);
    }
    .panel-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .panel-list { padding: 10px; }
    .panel-row {
      display: grid;
      grid-template-columns: 36px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--soft-line);
      border-radius: 8px;
      background: var(--panel);
      margin-bottom: 8px;
    }
    .panel-row:last-child { margin-bottom: 0; }
    .mini-rank {
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 7px;
      background: var(--green-soft);
      color: var(--green);
      font-weight: 900;
    }
    .mini-score {
      color: var(--green);
      font-weight: 900;
      padding: 6px 8px;
      border-radius: 999px;
      background: var(--green-soft);
    }
    .landing-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      padding: 10px;
      border-top: 1px solid var(--line);
      background: var(--panel-2);
    }
    .landing-stat {
      padding: 10px;
      border-radius: 7px;
      background: var(--panel);
      border: 1px solid var(--soft-line);
    }
    .landing-stat b { display: block; font-size: 18px; }
    .landing.hidden { display: none; }
    #appShell.hidden { display: none; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0; font-size: 23px; letter-spacing: 0; line-height: 1.1; }
    h2 { margin: 0 0 12px; font-size: 15px; letter-spacing: 0; }
    h3 { margin: 0 0 4px; font-size: 14px; letter-spacing: 0; }
    .subtle { color: var(--muted); }
    .eyebrow {
      color: var(--quiet);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .brand-mark {
      flex: 0 0 auto;
      width: 38px;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: inset 0 -1px 0 rgba(255, 255, 255, .08), var(--shadow);
      object-fit: contain;
      padding: 4px;
    }
    .header-right {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .status-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel-2);
      color: var(--muted);
      padding: 8px 11px;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .back-link {
      border-color: var(--line);
      background: var(--panel-2);
      color: var(--muted);
      min-height: 34px;
      padding: 7px 10px;
      font-size: 12px;
    }
    .theme-toggle {
      display: flex;
      gap: 3px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px;
      background: var(--panel-2);
    }
    .theme-toggle button {
      min-height: 30px;
      border: 0;
      border-radius: 999px;
      padding: 5px 10px;
      background: transparent;
      color: var(--muted);
      box-shadow: none;
      font-size: 12px;
    }
    .theme-toggle button.active {
      background: var(--panel);
      color: var(--green);
      box-shadow: var(--shadow);
    }
    .layout {
      display: grid;
      grid-template-columns: 352px minmax(0, 1fr);
      min-height: calc(100vh - 67px);
    }
    aside {
      padding: 18px;
      border-right: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      position: sticky;
      top: 67px;
      align-self: start;
      height: calc(100vh - 67px);
      overflow: auto;
    }
    main { min-width: 0; padding: 20px 24px 44px; }
    .section {
      border-bottom: 1px solid var(--line);
      padding: 0 0 18px;
      margin: 0 0 18px;
    }
    .section:last-child { border-bottom: 0; }
    label { display: block; margin: 12px 0 5px; font-weight: 700; font-size: 12px; color: var(--muted); }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      background: var(--field);
      color: var(--ink);
      font: inherit;
      transition: border-color .16s ease, box-shadow .16s ease, background .16s ease;
    }
    input:focus, textarea:focus, select:focus {
      outline: 0;
      border-color: var(--green);
      box-shadow: 0 0 0 3px var(--focus);
    }
    textarea { min-height: 78px; resize: vertical; }
    .checkgrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .checkgrid label {
      margin: 0;
      display: flex;
      gap: 7px;
      align-items: center;
      font-weight: 600;
      font-size: 13px;
      color: var(--ink);
    }
    .checkgrid input { width: auto; }
    .button-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    button {
      border: 1px solid var(--green);
      background: var(--green);
      color: var(--paper);
      border-radius: 7px;
      padding: 9px 13px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      min-height: 38px;
      transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
    }
    button.secondary { background: var(--panel); color: var(--green); }
    button.tertiary { background: var(--panel); border-color: var(--line); color: var(--ink); }
    button:hover { filter: brightness(.98); box-shadow: 0 10px 24px rgba(18, 28, 26, .10); }
    button:active { transform: translateY(1px); }
    .primary-wide { width: 100%; }
    .workspace-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 18px;
      margin-bottom: 16px;
    }
    .workspace-head h2 {
      margin: 4px 0 2px;
      font-size: 21px;
    }
    .workspace-actions {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .learning-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px 11px;
      min-height: 58px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 76px;
      box-shadow: var(--shadow);
    }
    .metric b { display: block; font-size: 21px; margin-top: 4px; }
    .metric span { font-size: 12px; font-weight: 750; color: var(--muted); }
    .tabs {
      display: flex;
      gap: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 4px;
      background: var(--panel);
      width: fit-content;
      max-width: 100%;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    .tab {
      background: transparent;
      border: 0;
      color: var(--muted);
      border-radius: 6px;
      padding: 8px 11px;
      min-height: 34px;
    }
    .tab.active { color: var(--green); background: var(--green-soft); }
    .opportunity {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      margin-bottom: 12px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }
    .opp-top {
      display: grid;
      grid-template-columns: 46px minmax(0, 1fr) 78px;
      gap: 14px;
      align-items: stretch;
      padding: 14px 14px 10px;
    }
    .rank {
      display: grid;
      place-items: center;
      min-height: 46px;
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--green);
      font-size: 17px;
      font-weight: 900;
    }
    .title { font-size: 15px; font-weight: 800; line-height: 1.3; }
    .score {
      text-align: center;
      border-radius: 7px;
      padding: 8px 6px;
      background: var(--green-soft);
      color: var(--green);
      font-weight: 800;
      min-height: 46px;
      display: grid;
      place-items: center;
    }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: var(--panel);
      white-space: nowrap;
    }
    .source-github, .source-gh_archive { border-color: color-mix(in srgb, var(--blue) 35%, var(--line)); color: var(--blue); background: var(--blue-soft); }
    .source-reddit { border-color: color-mix(in srgb, var(--coral) 35%, var(--line)); color: var(--coral); background: var(--coral-soft); }
    .source-hacker_news { border-color: color-mix(in srgb, var(--gold) 35%, var(--line)); color: var(--gold); background: var(--gold-soft); }
    .opp-body { padding: 0 14px 14px 74px; }
    .signal-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(86px, 1fr));
      gap: 8px;
      margin: 10px 0;
    }
    .signal {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px 8px;
      background: var(--panel-2);
    }
    .signal span { display: block; color: var(--quiet); font-size: 11px; font-weight: 750; }
    .signal b { display: block; font-size: 13px; margin-top: 1px; }
    .why { color: var(--muted); margin: 8px 0; }
    .action { border-left: 4px solid var(--green); padding: 9px 10px; background: var(--green-soft); margin-top: 8px; border-radius: 0 7px 7px 0; }
    .charts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .chart {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .bar-row {
      display: grid;
      grid-template-columns: 175px minmax(80px, 1fr) 58px;
      gap: 8px;
      align-items: center;
      margin: 8px 0;
    }
    .bar {
      height: 9px;
      border-radius: 999px;
      background: var(--soft-line);
      overflow: hidden;
    }
    .bar span { display: block; height: 100%; background: var(--blue); }
    .table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .table th, .table td { padding: 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    .table th { font-size: 12px; color: var(--muted); background: var(--panel-2); }
    a { color: var(--green); font-weight: 750; }
    .empty-state {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
      background: var(--panel);
    }
    .hidden { display: none; }
    @media (max-width: 900px) {
      .landing-main { grid-template-columns: 1fr; padding-top: 20px; }
      .hero-copy h1 { font-size: clamp(36px, 12vw, 58px); }
      .landing-stats { grid-template-columns: 1fr; }
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); position: static; height: auto; }
      .metrics, .charts { grid-template-columns: 1fr; }
      .workspace-head { grid-template-columns: 1fr; }
      .workspace-actions { justify-content: flex-start; }
      .opp-top { grid-template-columns: 42px minmax(0, 1fr); }
      .score { grid-column: 2; width: 78px; }
      .opp-body { padding-left: 14px; }
      .signal-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { align-items: flex-start; }
    }
  </style>
</head>
<body>
  <section class="landing" id="landingPage">
    <nav class="landing-nav">
      <div class="brand">
        <img class="brand-mark" src="/assets/EngageIQ-RadarScope-color.svg" alt="EngageIQ logo">
        <div>
          <div class="eyebrow">EngageIQ Platform</div>
          <strong>Opportunity intelligence for online engagement</strong>
        </div>
      </div>
      <div class="theme-toggle" aria-label="Theme">
        <button id="landingLightTheme" type="button" onclick="setTheme('light')">Light</button>
        <button id="landingDarkTheme" type="button" onclick="setTheme('dark')">Dark</button>
      </div>
    </nav>
    <div class="landing-main">
      <div class="hero-copy">
        <div class="eyebrow">Public signals. Personalized ranking. Faster decisions.</div>
        <h1>Find the best places to show up online.</h1>
        <p>EngageIQ scores GitHub, GH Archive, Reddit, and Hacker News-style opportunities against test personas or your own custom profile, then turns the best matches into ranked actions and a downloadable engagement brief.</p>
        <div class="hero-actions">
          <button onclick="enterDashboard()">Enter Dashboard</button>
        </div>
      </div>
      <div class="platform-panel">
        <div class="panel-top">
          <div>
            <div class="eyebrow">Live Demo View</div>
            <strong>Ranked opportunity feed</strong>
          </div>
          <span class="status-pill">10,500 records</span>
        </div>
        <div class="panel-list">
          <div class="panel-row">
            <div class="mini-rank">1</div>
            <div>
              <strong>Good first issue: NLP pipeline contribution</strong>
              <div class="subtle">Sofia ML Student · GitHub · 45 min</div>
            </div>
            <div class="mini-score">91.4</div>
          </div>
          <div class="panel-row">
            <div class="mini-rank">2</div>
            <div>
              <strong>Kubernetes discussion with expert-comment gap</strong>
              <div class="subtle">David DevOps Engineer · Reddit · 30 min</div>
            </div>
            <div class="mini-score">88.7</div>
          </div>
          <div class="panel-row">
            <div class="mini-rank">3</div>
            <div>
              <strong>Rising developer tool gaining velocity</strong>
              <div class="subtle">Lina Trend Spotter · GH Archive · 60 min</div>
            </div>
            <div class="mini-score">84.9</div>
          </div>
        </div>
        <div class="landing-stats">
          <div class="landing-stat"><span class="subtle">Sources</span><b>4</b></div>
          <div class="landing-stat"><span class="subtle">Domains</span><b>15</b></div>
          <div class="landing-stat"><span class="subtle">Exports</span><b>CSV/PDF</b></div>
        </div>
      </div>
    </div>
  </section>
  <div id="appShell" class="hidden">
  <header>
    <div class="brand">
      <img class="brand-mark" src="/assets/EngageIQ-RadarScope-color.svg" alt="EngageIQ logo">
      <div>
        <div class="eyebrow">BAX-423 Final Build</div>
        <h1>EngageIQ</h1>
        <div class="subtle">Smart engagement opportunity scorer for GitHub, GH Archive, Reddit, and Hacker News.</div>
      </div>
    </div>
    <div class="header-right">
      <button class="back-link" type="button" onclick="showLanding()">Back to Landing</button>
      <div class="status-pill" id="datasetStatus">Loading dataset...</div>
      <div class="theme-toggle" aria-label="Theme">
        <button id="lightTheme" type="button" onclick="setTheme('light')">Light</button>
        <button id="darkTheme" type="button" onclick="setTheme('dark')">Dark</button>
      </div>
    </div>
  </header>
  <div class="layout">
    <aside>
      <div class="section">
        <div class="eyebrow">Configuration</div>
        <h2>User Profile</h2>
        <label for="persona">Test persona</label>
        <select id="persona"></select>
        <div class="button-row">
          <button class="secondary" type="button" onclick="saveCurrentProfile()">Save Profile</button>
          <button class="tertiary" type="button" onclick="deleteSelectedProfile()">Delete Saved</button>
        </div>
        <p class="subtle" id="profileStatus"></p>
        <label for="profileName">Name</label>
        <input id="profileName" placeholder="Your name or another person's name">
        <label for="interests">Interests and skills</label>
        <textarea id="interests"></textarea>
        <label for="skillsets">Skillsets</label>
        <textarea id="skillsets" placeholder="Python, Kubernetes, technical writing, APIs, community building"></textarea>
        <label for="goal">Goal</label>
        <textarea id="goal"></textarea>
        <label for="avoid">Avoid</label>
        <input id="avoid">
        <label for="timeBudget">Time budget, hours/week</label>
        <input id="timeBudget" type="number" min="1" max="20" step="1">
        <label>Platforms</label>
        <div class="checkgrid" id="platforms"></div>
        <div class="button-row">
          <button class="primary-wide" onclick="recommend()">Rank Opportunities</button>
          <button class="secondary primary-wide" onclick="simulateLearning()">Simulate Feedback</button>
        </div>
      </div>
      <div class="section">
        <div class="eyebrow">Deliverable</div>
        <h2>Brief Export</h2>
        <div class="button-row">
          <button class="secondary" onclick="exportBrief('csv')">CSV</button>
          <button class="secondary" onclick="exportBrief('pdf')">PDF</button>
        </div>
        <p class="subtle" id="exportStatus"></p>
      </div>
      <div class="section">
        <div class="eyebrow">Adaptive Ranking</div>
        <h2>Learning Result</h2>
        <div class="learning-box">
          <p class="subtle" id="learningStatus">Run the simulation to show measurable ranking improvement.</p>
        </div>
      </div>
    </aside>
    <main>
      <div class="workspace-head">
        <div>
          <div class="eyebrow">Recommendation Workspace</div>
          <h2 id="workspaceTitle">Ranked Opportunities</h2>
          <div class="subtle" id="workspaceSubtitle">Choose a persona and rank public engagement opportunities.</div>
        </div>
        <div class="workspace-actions">
          <button class="secondary" onclick="recommend()">Refresh Rank</button>
          <button class="tertiary" onclick="showTab('analytics')">View Trends</button>
        </div>
      </div>
      <div class="metrics" id="metrics"></div>
      <div class="tabs">
        <button class="tab active" onclick="showTab('recommendations')">Recommendations</button>
        <button class="tab" onclick="showTab('analytics')">Trend Analytics</button>
        <button class="tab" onclick="showTab('personaTests')">Persona Coverage</button>
      </div>
      <section id="recommendations"></section>
      <section id="analytics" class="hidden"></section>
      <section id="personaTests" class="hidden"></section>
    </main>
  </div>
  </div>
  <script>
    let bootstrap = null;
    let lastRecommendations = [];
    let recommendTimer = null;
    const savedProfilesKey = 'engageiq-saved-profiles';

    async function init() {
      applyTheme(savedTheme);
      bootstrap = await fetch('/api/bootstrap').then(r => r.json());
      const backendLabel = bootstrap.embedding?.backend === 'sbert' ? 'SBERT dense embeddings' : `${bootstrap.embedding?.backend || 'vector'} retrieval`;
      document.getElementById('datasetStatus').textContent =
        `${bootstrap.summary.records.toLocaleString()} records · ${bootstrap.summary.domains} domains · ${backendLabel}`;
      document.getElementById('learningStatus').textContent =
        `${bootstrap.feedback?.total_events || 0} saved feedback events loaded. Run the simulation to show measured adaptation.`;
      renderPersonaControls();
      renderMetrics();
      renderAnalytics();
      renderPersonaTests();
      recommend();
    }

    function renderPersonaControls() {
      const persona = document.getElementById('persona');
      populatePersonaDropdown();
      const platforms = document.getElementById('platforms');
      platforms.innerHTML = bootstrap.sources.map(s =>
        `<label><input type="checkbox" value="${s}" checked> ${s.replace('_', ' ')}</label>`).join('');
      persona.addEventListener('change', () => {
        fillPersona();
        scheduleRecommend(0);
      });
      ['profileName', 'interests', 'skillsets', 'goal', 'avoid', 'timeBudget'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => scheduleRecommend(500));
      });
      platforms.addEventListener('change', () => scheduleRecommend(0));
      fillPersona();
    }

    function populatePersonaDropdown(selectedValue) {
      const persona = document.getElementById('persona');
      const current = selectedValue || persona.value || 'Sofia';
      const saved = getSavedProfiles();
      const savedOptions = saved.length
        ? `<optgroup label="Saved custom profiles">${saved.map(profile =>
            `<option value="saved:${profile.id}">${escapeHtml(profile.name)}</option>`
          ).join('')}</optgroup>`
        : '';
      persona.innerHTML = `
        <optgroup label="Test personas">
          ${Object.entries(bootstrap.personas).map(([key, val]) => `<option value="${key}">${escapeHtml(val.name)}</option>`).join('')}
        </optgroup>
        <optgroup label="Create">
          <option value="custom">Custom profile</option>
        </optgroup>
        ${savedOptions}
      `;
      if ([...persona.options].some(option => option.value === current)) {
        persona.value = current;
      }
    }

    function fillPersona() {
      const key = document.getElementById('persona').value;
      if (key.startsWith('saved:')) {
        const profile = getSavedProfiles().find(item => `saved:${item.id}` === key);
        if (!profile) {
          document.getElementById('persona').value = 'custom';
          fillPersona();
          return;
        }
        hydrateProfile(profile);
        document.getElementById('profileStatus').textContent = `Loaded saved profile: ${profile.name}`;
        return;
      }
      if (key === 'custom') {
        document.getElementById('profileName').value = '';
        document.getElementById('interests').value = '';
        document.getElementById('skillsets').value = '';
        document.getElementById('goal').value = '';
        document.getElementById('avoid').value = '';
        document.getElementById('timeBudget').value = 4;
        document.querySelectorAll('#platforms input').forEach(cb => cb.checked = true);
        document.getElementById('workspaceTitle').textContent = 'Ranked Opportunities for Custom Profile';
        document.getElementById('workspaceSubtitle').textContent = 'Enter a name, interests, goals, and skillsets to score a custom profile.';
        document.getElementById('profileStatus').textContent = 'Create a new custom profile or save the current inputs.';
        return;
      }
      const data = bootstrap.personas[key];
      document.getElementById('profileName').value = data.name;
      document.getElementById('interests').value = data.interests;
      document.getElementById('skillsets').value = extractSkillsets(data.interests);
      document.getElementById('goal').value = data.goal;
      document.getElementById('avoid').value = data.avoid || '';
      document.getElementById('timeBudget').value = data.time_budget;
      document.querySelectorAll('#platforms input').forEach(cb => cb.checked = data.platforms.includes(cb.value));
      document.getElementById('profileStatus').textContent = 'Test personas are fixed; edit fields and click Save Profile to store your own version.';
    }

    function profilePayload() {
      const key = document.getElementById('persona').value;
      const fallback = bootstrap.personas[key] || {};
      return {
        persona: key,
        name: document.getElementById('profileName').value || fallback.name || 'Custom Profile',
        interests: document.getElementById('interests').value,
        skillsets: document.getElementById('skillsets').value,
        goal: document.getElementById('goal').value,
        avoid: document.getElementById('avoid').value,
        time_budget: Number(document.getElementById('timeBudget').value || 4),
        platforms: [...document.querySelectorAll('#platforms input:checked')].map(cb => cb.value),
        limit: 10
      };
    }

    function hydrateProfile(profile) {
      document.getElementById('profileName').value = profile.name || '';
      document.getElementById('interests').value = profile.interests || '';
      document.getElementById('skillsets').value = profile.skillsets || '';
      document.getElementById('goal').value = profile.goal || '';
      document.getElementById('avoid').value = profile.avoid || '';
      document.getElementById('timeBudget').value = profile.time_budget || 4;
      document.querySelectorAll('#platforms input').forEach(cb => cb.checked = (profile.platforms || bootstrap.sources).includes(cb.value));
    }

    function getSavedProfiles() {
      try {
        const saved = JSON.parse(localStorage.getItem(savedProfilesKey) || '[]');
        return Array.isArray(saved) ? saved : [];
      } catch {
        return [];
      }
    }

    function setSavedProfiles(profiles) {
      localStorage.setItem(savedProfilesKey, JSON.stringify(profiles));
    }

    function saveCurrentProfile() {
      const payload = profilePayload();
      if (!payload.name || payload.name === 'Custom Profile') {
        document.getElementById('profileStatus').textContent = 'Add a name before saving this profile.';
        return;
      }
      if (!payload.interests.trim() && !payload.skillsets.trim() && !payload.goal.trim()) {
        document.getElementById('profileStatus').textContent = 'Add interests, skillsets, or a goal before saving.';
        return;
      }
      const selected = document.getElementById('persona').value;
      const existingId = selected.startsWith('saved:') ? selected.slice(6) : slugId(payload.name);
      const profile = {
        id: existingId,
        name: payload.name,
        interests: payload.interests,
        skillsets: payload.skillsets,
        goal: payload.goal,
        avoid: payload.avoid,
        time_budget: payload.time_budget,
        platforms: payload.platforms,
        saved_at: new Date().toISOString()
      };
      const profiles = getSavedProfiles().filter(item => item.id !== profile.id);
      profiles.push(profile);
      profiles.sort((a, b) => a.name.localeCompare(b.name));
      setSavedProfiles(profiles);
      populatePersonaDropdown(`saved:${profile.id}`);
      document.getElementById('profileStatus').textContent = `Saved profile: ${profile.name}`;
      scheduleRecommend(0);
    }

    function deleteSelectedProfile() {
      const selected = document.getElementById('persona').value;
      if (!selected.startsWith('saved:')) {
        document.getElementById('profileStatus').textContent = 'Select a saved custom profile to delete.';
        return;
      }
      const id = selected.slice(6);
      const removed = getSavedProfiles().find(item => item.id === id);
      setSavedProfiles(getSavedProfiles().filter(item => item.id !== id));
      populatePersonaDropdown('custom');
      fillPersona();
      document.getElementById('profileStatus').textContent = removed ? `Deleted saved profile: ${removed.name}` : 'Deleted saved profile.';
      scheduleRecommend(0);
    }

    function slugId(name) {
      const base = String(name || 'profile').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'profile';
      let id = base;
      let index = 2;
      const existing = new Set(getSavedProfiles().map(item => item.id));
      while (existing.has(id)) {
        id = `${base}-${index}`;
        index += 1;
      }
      return id;
    }

    function enterDashboard(mode) {
      document.getElementById('landingPage').classList.add('hidden');
      document.getElementById('appShell').classList.remove('hidden');
      if (mode === 'custom') {
        document.getElementById('persona').value = 'custom';
        fillPersona();
      }
      setTimeout(() => recommend(), 50);
    }

    function showLanding() {
      document.getElementById('appShell').classList.add('hidden');
      document.getElementById('landingPage').classList.remove('hidden');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    async function recommend() {
      const container = document.getElementById('recommendations');
      if (document.getElementById('appShell').classList.contains('hidden')) return;
      if (!hasEnoughProfileInput()) {
        document.getElementById('workspaceTitle').textContent = 'Ranked Opportunities for Custom Profile';
        document.getElementById('workspaceSubtitle').textContent = 'Enter interests, goals, or skillsets to score a custom profile.';
        container.innerHTML = '<div class="empty-state">Add custom profile details, then EngageIQ will rank matching opportunities.</div>';
        return;
      }
      container.innerHTML = '<div class="empty-state">Ranking opportunities...</div>';
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(profilePayload())
      }).then(r => r.json());
      lastRecommendations = res.recommendations;
      document.getElementById('workspaceTitle').textContent = `Ranked Opportunities for ${res.profile.name}`;
      document.getElementById('workspaceSubtitle').textContent =
        `${res.recommendations.length} deterministic recommendations filtered across ${res.profile.platforms.map(p => p.replace('_', ' ')).join(', ')}. Last ranked ${new Date().toLocaleTimeString()}.`;
      renderRecommendations(res.recommendations);
    }

    function scheduleRecommend(delay = 300) {
      clearTimeout(recommendTimer);
      recommendTimer = setTimeout(() => recommend(), delay);
    }

    function hasEnoughProfileInput() {
      if (document.getElementById('persona').value !== 'custom') return true;
      return Boolean(
        document.getElementById('interests').value.trim()
        || document.getElementById('skillsets').value.trim()
        || document.getElementById('goal').value.trim()
      );
    }

    function renderRecommendations(items) {
      const container = document.getElementById('recommendations');
      if (!items.length) {
        container.innerHTML = '<div class="empty-state">No recommendations matched the current filters.</div>';
        return;
      }
      container.innerHTML = items.map((item, idx) => `
        <article class="opportunity">
          <div class="opp-top">
            <div class="rank">${idx + 1}</div>
            <div>
              <div class="title">${escapeHtml(item.title)}</div>
              <div class="badges">
                <span class="badge source-${item.source}">${item.source.replace('_', ' ')}</span>
                <span class="badge">${escapeHtml(item.domain)}</span>
                <span class="badge">${escapeHtml(item.community)}</span>
                <span class="badge">${item.effort_minutes} min</span>
                ${item.good_first_issue ? '<span class="badge">good first issue</span>' : ''}
              </div>
            </div>
            <div class="score">${Number(item.diversified_score).toFixed(1)}</div>
          </div>
          <div class="opp-body">
            <div class="signal-grid">
              <div class="signal"><span>Activity</span><b>${formatScore(item.activity)}</b></div>
              <div class="signal"><span>Health</span><b>${formatScore(item.health)}</b></div>
              <div class="signal"><span>Visibility</span><b>${formatScore(item.visibility)}</b></div>
              <div class="signal"><span>Growth</span><b>${formatScore(item.growth_rate)}</b></div>
            </div>
            <p class="why">${escapeHtml(item.why_this)}</p>
            <div class="action">${escapeHtml(item.suggested_action)}</div>
            <div class="button-row">
              <button class="tertiary" onclick="feedback('${item.id}', 'engage')">Engage</button>
              <button class="tertiary" onclick="feedback('${item.id}', 'bookmark')">Bookmark</button>
              <button class="tertiary" onclick="feedback('${item.id}', 'skip')">Skip</button>
            </div>
          </div>
        </article>
      `).join('');
    }

    async function feedback(id, action) {
      const result = await fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...profilePayload(), id, action})
      }).then(r => r.json());
      if (result.ok) {
        document.getElementById('learningStatus').textContent =
          `${result.feedback.total_events} saved feedback events. Last action: ${action}. Ranking weights updated and persisted.`;
      }
      recommend();
    }

    async function simulateLearning() {
      const result = await fetch('/api/simulate-learning', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(profilePayload())
      }).then(r => r.json());
      document.getElementById('learningStatus').textContent =
        `Precision@10 improved from ${result.precision_at_10_before} to ${result.precision_at_10_after} ` +
        `over ${result.rounds} rounds. Actions: ${JSON.stringify(result.actions)}.`;
      recommend();
    }

    async function exportBrief(type) {
      const result = await fetch('/api/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...profilePayload(), type})
      }).then(r => r.json());
      document.getElementById('exportStatus').innerHTML = `<a href="${result.url}">Download ${type.toUpperCase()} brief</a>`;
    }

    function renderMetrics() {
      const s = bootstrap.summary;
      document.getElementById('metrics').innerHTML = [
        ['Records', s.records.toLocaleString()],
        ['Sources', s.sources.length],
        ['Domains', s.domains],
        ['Avg activity', s.avg_activity],
        ['Max growth', s.max_growth]
      ].map(([label, value]) => `<div class="metric"><span class="subtle">${label}</span><b>${value}</b></div>`).join('');
    }

    function renderAnalytics() {
      const trends = bootstrap.trends;
      const maxDomain = Math.max(...trends.domains.map(d => d.records));
      const domainBars = trends.domains.map(d => `
        <div class="bar-row">
          <span>${escapeHtml(d.domain)}</span>
          <div class="bar"><span style="width:${(d.records / maxDomain) * 100}%"></span></div>
          <span>${d.records}</span>
        </div>`).join('');
      const communities = trends.communities.map(c => `
        <tr><td>${escapeHtml(c.source)}</td><td>${escapeHtml(c.community)}</td><td>${c.records}</td><td>${Number(c.avg_growth).toFixed(2)}</td></tr>`
      ).join('');
      document.getElementById('analytics').innerHTML = `
        <div class="charts">
          <div class="chart"><h2>Domain Volume</h2>${domainBars}</div>
          <div class="chart">
            <h2>Fast Communities</h2>
            <table class="table"><thead><tr><th>Source</th><th>Community</th><th>Records</th><th>Growth</th></tr></thead><tbody>${communities}</tbody></table>
          </div>
        </div>`;
    }

    function renderPersonaTests() {
      const rows = [
        ['Sofia', 'Top-10 GitHub + ML focus', 'Uses good-first-issue boost, Python/ML relevance, C++ avoid filter'],
        ['Emma', 'Beginner career-switcher fit', 'Uses beginner boost, approachable Python/web signals, and advanced-systems avoid filter'],
        ['David', 'Kubernetes/infra focus', 'DevOps/K8s and Cloud APIs receive ranking boost; general frontend avoided'],
        ['Lina', 'Recency and velocity', 'Trend persona receives growth and freshness boost plus analytics view'],
        ['Raj', 'Developer-tool relevance', 'Developer Tools/API/CLI text drives embeddings; skips feed adaptive weights']
      ].map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>Ready for demo validation</td></tr>`).join('');
      document.getElementById('personaTests').innerHTML = `
        <table class="table">
          <thead><tr><th>Persona</th><th>Pass Criterion</th><th>Implementation Hook</th><th>Status</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    function showTab(name) {
      ['recommendations', 'analytics', 'personaTests'].forEach(id => document.getElementById(id).classList.toggle('hidden', id !== name));
      document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));
      [...document.querySelectorAll('.tab')].find(btn => btn.textContent.toLowerCase().includes(name === 'personaTests' ? 'persona' : name.slice(0, 8))).classList.add('active');
    }

    function setTheme(theme) {
      localStorage.setItem('engageiq-theme', theme);
      applyTheme(theme);
    }

    function applyTheme(theme) {
      document.documentElement.dataset.theme = theme;
      document.getElementById('lightTheme')?.classList.toggle('active', theme === 'light');
      document.getElementById('darkTheme')?.classList.toggle('active', theme === 'dark');
      document.getElementById('landingLightTheme')?.classList.toggle('active', theme === 'light');
      document.getElementById('landingDarkTheme')?.classList.toggle('active', theme === 'dark');
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }

    function formatScore(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return 'n/a';
      return n.toFixed(2);
    }

    function extractSkillsets(text) {
      return String(text || '')
        .split(',')
        .map(part => part.trim())
        .filter(Boolean)
        .slice(0, 6)
        .join(', ');
    }

    document.addEventListener('pointermove', event => {
      document.body.style.setProperty('--cursor-x', `${event.clientX}px`);
      document.body.style.setProperty('--cursor-y', `${event.clientY}px`);
    });

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), EngageIQHandler)
    print(f"EngageIQ running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
