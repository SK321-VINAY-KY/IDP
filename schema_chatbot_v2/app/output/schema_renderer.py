"""
File: schema_renderer.py
Purpose: Renders a confirmed schema record to bytes — JSON (pretty-printed)
         or PDF (formatted table via reportlab). Called only by the download
         endpoints in api/routes.py; never by the conversation manager.
Owner: engineer-a@idp-pilot
Created: 2026-09-01 | Deps: reportlab (PDF), stdlib json (JSON)
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------

def render_json(record: Dict[str, Any]) -> bytes:
    """
    Returns the full schema record as indented UTF-8 JSON bytes.
    The record already contains schema_id, document_type, confirmed_at,
    session_id, turn_count_at_confirm, and the nested schema dict.
    """
    return json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# PDF renderer
# ---------------------------------------------------------------------------

def render_pdf(record: Dict[str, Any]) -> bytes:
    """
    Builds a human-readable PDF of the confirmed schema using reportlab.
    Layout:
        - Header with document_type, schema_id, confirmed_at
        - One table row per field: name | type | required | notes
        - Metadata footer

    Returns raw PDF bytes (application/pdf).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles   = getSampleStyleSheet()
    title_s  = styles["Title"]
    heading2 = styles["Heading2"]
    normal   = styles["Normal"]
    small    = ParagraphStyle("small", parent=normal, fontSize=8, textColor=colors.grey)
    code_s   = ParagraphStyle("code", parent=normal, fontName="Courier", fontSize=9)

    schema        = record.get("schema", {})
    doc_type      = schema.get("document_type") or record.get("document_type") or "—"
    schema_id     = record.get("schema_id", "—")
    confirmed_at  = record.get("confirmed_at", "—")
    session_id    = record.get("session_id", "—")
    turn_count    = record.get("turn_count_at_confirm", "—")
    fields: List[Dict[str, Any]] = schema.get("fields", [])

    story = []

    # ── Title ────────────────────────────────────────────────────────────
    story.append(Paragraph("IDP — Confirmed Extraction Schema", title_s))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB")))
    story.append(Spacer(1, 0.4 * cm))

    # ── Meta block ───────────────────────────────────────────────────────
    meta_rows = [
        ["Document type",   doc_type],
        ["Schema ID",       schema_id],
        ["Confirmed at",    confirmed_at],
        ["Session ID",      session_id],
        ["Turns to confirm", str(turn_count)],
        ["Total fields",    str(len(fields))],
    ]
    meta_table = Table(meta_rows, colWidths=[4.5 * cm, None])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#374151")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.6 * cm))

    # ── Fields table ─────────────────────────────────────────────────────
    story.append(Paragraph("Extraction Fields", heading2))
    story.append(Spacer(1, 0.2 * cm))

    if not fields:
        story.append(Paragraph("(no fields defined)", normal))
    else:
        header = ["Field name", "Type", "Required", "Notes"]
        rows   = [header]
        for f in fields:
            name     = str(f.get("name", ""))
            ftype    = str(f.get("type") or "—")
            req      = "Yes" if f.get("required") else "No"

            # Notes: collect extra metadata inline
            note_parts = []
            if ftype == "array" and f.get("item_type"):
                note_parts.append(f"items: {f['item_type']}")
            if f.get("currency"):
                note_parts.append(f"currency: {f['currency']}")
            if f.get("pattern"):
                note_parts.append(f"pattern: {f['pattern']}")
            if f.get("description"):
                note_parts.append(f['description'])
            if ftype == "object" and f.get("fields"):
                sub = ", ".join(str(k) for k in f["fields"])
                note_parts.append(f"sub-fields: {sub}")
            notes = "; ".join(note_parts) if note_parts else "—"

            rows.append([
                Paragraph(name, code_s),
                Paragraph(ftype, normal),
                Paragraph(req, normal),
                Paragraph(notes, small),
            ])

        col_widths = [5 * cm, 2.5 * cm, 2 * cm, None]
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            # Header row
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9),
            ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
            # Data rows
            ("FONTSIZE",    (0, 1), (-1, -1), 8),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)

    # ── Footer ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.2 * cm))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(
        f"Generated by IDP Pipeline · schema_chatbot_v2 · {generated_at}",
        small,
    ))

    doc.build(story)
    return buf.getvalue()
