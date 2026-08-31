"""
File: job_pdf_report.py
Purpose: Generates a styled, readable PDF summary and extraction report
         from a pipeline job JSON dictionary using ReportLab.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_job_pdf(job_data: Dict[str, Any]) -> bytes:
    """
    Builds a multi-page PDF document from the full pipeline job result.
    Returns the PDF content as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'BodyTextDark',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155'),
    )
    key_style = ParagraphStyle(
        'KeyText',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor('#1e293b'),
    )
    val_style = ParagraphStyle(
        'ValText',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a'),
    )

    story = []

    # Title & Header
    job_id = job_data.get("job_id", "Unknown Job")
    status = job_data.get("status", "unknown").upper()
    story.append(Paragraph(f"IDP Pipeline Execution Report", title_style))
    story.append(Paragraph(f"Job ID: <b>{job_id}</b> &nbsp;|&nbsp; Status: <b>{status}</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=12))

    # Summary Table
    created_at = job_data.get("created_at", "--")
    finished_at = job_data.get("finished_at", "--")
    wall_time = f"{job_data.get('wall_time_s', '--')}s" if job_data.get('wall_time_s') is not None else "--"
    schema_id = job_data.get("schema_id", "--")
    successes = job_data.get("successes", [])
    failures = job_data.get("failures", [])
    total_docs = len(job_data.get("targets", [])) or (len(successes) + len(failures))

    summary_data = [
        [
            Paragraph("<b>Created At:</b>", key_style), Paragraph(str(created_at), val_style),
            Paragraph("<b>Finished At:</b>", key_style), Paragraph(str(finished_at), val_style),
        ],
        [
            Paragraph("<b>Schema ID:</b>", key_style), Paragraph(str(schema_id), val_style),
            Paragraph("<b>Wall Time:</b>", key_style), Paragraph(wall_time, val_style),
        ],
        [
            Paragraph("<b>Total Targets:</b>", key_style), Paragraph(str(total_docs), val_style),
            Paragraph("<b>Success / Failed:</b>", key_style),
            Paragraph(f"<font color='#059669'><b>{len(successes)} ok</b></font> / <font color='#dc2626'><b>{len(failures)} failed</b></font>", val_style),
        ],
    ]

    t_summary = Table(summary_data, colWidths=[90, 180, 90, 180])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))

    # Successful Document Extractions
    if successes:
        story.append(Paragraph(f"Successful Document Extractions ({len(successes)})", h2_style))
        for idx, s in enumerate(successes, 1):
            doc_flowables = []
            pdf_name = s.get("pdf", "Unknown Doc")
            pages = s.get("pages", 1)
            conf = f"{s.get('avg_conf', 0.0):.3f}"
            elapsed = f"{s.get('elapsed_s', 0)}s"
            extract_elapsed = f"{s.get('extract_elapsed_s', 0)}s"
            db_run_id = s.get("db_run_id")

            doc_header_text = (
                f"<b>{idx}. {pdf_name}</b> &nbsp;&nbsp;"
                f"<font color='#64748b' size='8'>(Pages: {pages} | Confidence: {conf} | "
                f"L1+2: {elapsed} | L3: {extract_elapsed}"
                f"{f' | PG Run #{db_run_id}' if db_run_id else ''})</font>"
            )
            doc_flowables.append(Paragraph(doc_header_text, body_style))
            doc_flowables.append(Spacer(1, 4))

            ext_data = s.get("extracted_data")
            if ext_data and isinstance(ext_data, dict):
                field_rows = [
                    [Paragraph("<b>Field Name</b>", key_style), Paragraph("<b>Extracted Value</b>", key_style)]
                ]
                for k, v in ext_data.items():
                    val_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    field_rows.append([
                        Paragraph(f"<code>{k}</code>", key_style),
                        Paragraph(val_str.replace("<", "&lt;").replace(">", "&gt;"), val_style),
                    ])

                t_fields = Table(field_rows, colWidths=[150, 390])
                t_fields.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2fe')),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#bae6fd')),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ]))
                doc_flowables.append(t_fields)
            else:
                doc_flowables.append(Paragraph("<i>No structured field extraction data available.</i>", val_style))

            doc_flowables.append(Spacer(1, 10))
            story.append(KeepTogether(doc_flowables))

    # Failures Section (if any)
    if failures:
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Failed Documents ({len(failures)})", h2_style))
        fail_rows = [
            [Paragraph("<b>Document</b>", key_style), Paragraph("<b>Error Type</b>", key_style), Paragraph("<b>Error Details</b>", key_style)]
        ]
        for f in failures:
            fail_rows.append([
                Paragraph(str(f.get("pdf", "--")), key_style),
                Paragraph(str(f.get("error_type", "Error")), val_style),
                Paragraph(str(f.get("error", "--")), val_style),
            ])
        t_fail = Table(fail_rows, colWidths=[140, 110, 290])
        t_fail.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fee2e2')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#fca5a5')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fef2f2')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_fail)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
