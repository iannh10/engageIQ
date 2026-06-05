"""Export helpers for engagement briefs."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception:  # pragma: no cover
    colors = None
    letter = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None


def export_csv(frame: pd.DataFrame, path: Path) -> Path:
    fields = ["rank", "source", "domain", "title", "diversified_score", "effort_minutes", "why_this", "suggested_action", "url"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy().reset_index(drop=True)
    output.insert(0, "rank", range(1, len(output) + 1))
    output[fields].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def export_pdf(frame: pd.DataFrame, path: Path, profile_name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if SimpleDocTemplate is None:
        write_minimal_pdf(path, frame, profile_name)
        return path

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = [
        Paragraph("EngageIQ Weekly Engagement Brief", styles["Title"]),
        Paragraph(f"Profile: {profile_name}", styles["Normal"]),
        Spacer(1, 12),
    ]
    data = [["Rank", "Source", "Domain", "Opportunity", "Score", "Action"]]
    for rank, (_, row) in enumerate(frame.head(10).iterrows(), 1):
        data.append(
            [
                str(rank),
                str(row["source"]),
                str(row["domain"]),
                Paragraph(str(row["title"]), styles["BodyText"]),
                f"{float(row['diversified_score']):.1f}",
                Paragraph(str(row["suggested_action"]), styles["BodyText"]),
            ]
        )
    table = Table(data, colWidths=[28, 58, 88, 160, 42, 160])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4f46")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d6dedb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8f7")]),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return path


def write_minimal_pdf(path: Path, frame: pd.DataFrame, profile_name: str) -> None:
    """Write a basic one-page PDF using only the standard library."""

    lines = ["EngageIQ Weekly Engagement Brief", f"Profile: {profile_name}", ""]
    for rank, (_, row) in enumerate(frame.head(10).iterrows(), 1):
        title = str(row["title"])[:88]
        score = f"{float(row['diversified_score']):.1f}"
        action = str(row["suggested_action"])[:96]
        lines.extend([f"{rank}. [{row['source']}] {row['domain']} | score {score}", title, action, ""])

    y = 760
    commands = ["BT", "/F1 11 Tf"]
    for line in lines[:44]:
        safe = line.encode("ascii", "ignore").decode("ascii").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"54 {y} Td ({safe}) Tj")
        commands.append(f"-54 -15 Td")
        y -= 15
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    content = [b"%PDF-1.4\n"]
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(sum(len(part) for part in content))
        content.append(f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_at = sum(len(part) for part in content)
    xref = [f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    xref.append(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii"))
    path.write_bytes(b"".join(content + xref))
