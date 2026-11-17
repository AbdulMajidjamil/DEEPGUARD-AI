# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
"""PDF forensic report for a completed analysis (reportlab, no external services)."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from config import APP_DEVELOPER, APP_FULL_NAME, DISCLAIMER


def _data_uri_image(uri: str, width: float) -> Image | None:
    try:
        raw = base64.b64decode(uri.split(",", 1)[1])
        img = Image(io.BytesIO(raw))
        ratio = img.imageHeight / float(img.imageWidth)
        img.drawWidth, img.drawHeight = width, width * ratio
        return img
    except Exception:
        return None


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def build_pdf(job: dict[str, Any]) -> bytes:
    """Render a completed (public) job dict to PDF bytes."""
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#0b2447"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#19376d"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="DEEPGUARD AI forensic report", author=APP_DEVELOPER,
    )
    story: list[Any] = [
        _p("DEEPGUARD AI - Forensic Analysis Report", h1),
        _p(APP_FULL_NAME, small),
        Spacer(1, 6),
    ]

    if job.get("is_demo"):
        warn = Table(
            [[_p("DEMO MODE - no trained checkpoint was loaded. The verdict below comes from a rule-based "
                 "temporal heuristic, NOT a trained deepfake classifier. Do not rely on it.", body)]],
            colWidths=[174 * mm],
        )
        warn.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#b8860b")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story += [warn, Spacer(1, 8)]

    video = job.get("video") or {}
    completed = job.get("completed_at")
    rows = [
        ("Task ID", job.get("task_id", "")),
        ("File", job.get("filename", "")),
        ("Verdict", job.get("label", "")),
        ("Confidence (in verdict)", f"{float(job.get('confidence', 0)) * 100:.1f}%"),
        ("Manipulation probability", f"{float(job.get('manipulation_probability', 0)) * 100:.1f}%"),
        ("Inference mode", job.get("inference_mode", "")),
        ("Model ID", job.get("model_id", "")),
        ("Device", job.get("device", "")),
        ("Frames sampled / with face", f"{job.get('frames_sampled', '?')} / {job.get('faces_detected', '?')}"),
        ("Max faces in one frame", job.get("max_faces_in_frame", "")),
        ("Video", f"{video.get('width', '?')}x{video.get('height', '?')} @ {video.get('fps', '?')} fps, "
                  f"{video.get('duration', '?')} s"),
        ("Completed", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(completed)) if completed else ""),
    ]
    table = Table([[_p(k, body), _p(v, body)] for k, v in rows], colWidths=[55 * mm, 119 * mm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d0e0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2fb")),
    ]))
    story += [table, Spacer(1, 8)]

    persons = job.get("persons") or []
    if persons:
        story.append(_p("Per-person results", h2))
        data = [[_p("Track", body), _p("Label", body), _p("P(fake)", body), _p("Faces analysed", body)]]
        for person in persons:
            data.append([
                _p(person.get("track_id"), body), _p(person.get("label"), body),
                _p(f"{float(person.get('probability_fake', 0)) * 100:.1f}%", body),
                _p(person.get("faces_analysed"), body),
            ])
        t2 = Table(data, colWidths=[30 * mm, 30 * mm, 40 * mm, 40 * mm])
        t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d0e0"))]))
        story += [t2, Spacer(1, 8)]

    frames = [f for f in (job.get("suspicious_frames") or []) if f.get("thumb")]
    if frames:
        story.append(_p("Suspicious frames (highest anomaly index)", h2))
        cells = []
        for f in frames[:6]:
            img = _data_uri_image(f["thumb"], 40 * mm)
            caption = _p(f"frame {f.get('index')} @ {f.get('timestamp')} s, anomaly {f.get('anomaly')}", small)
            cells.append([img, caption] if img else [caption])
        # lay out three per row
        grid_rows, row = [], []
        for cell in cells:
            row.append(cell)
            if len(row) == 3:
                grid_rows.append(row)
                row = []
        if row:
            grid_rows.append(row + [""] * (3 - len(row)))
        story.append(Table(grid_rows, colWidths=[58 * mm] * 3))
        story.append(_p("The anomaly index marks where temporal artefacts concentrate. It is a lead for a human "
                        "reviewer, not a classifier output.", small))
        story.append(Spacer(1, 8))

    warnings = job.get("warnings") or []
    if warnings:
        story.append(_p("Warnings", h2))
        story += [_p(f"- {w}", body) for w in warnings]
        story.append(Spacer(1, 6))

    story += [_p("Limitations", h2), _p(job.get("disclaimer") or DISCLAIMER, body), Spacer(1, 6),
              _p(f"Generated by DEEPGUARD AI v{job.get('app_version', '')} - {APP_DEVELOPER}. "
                 "Derived from SENTINEL by Dev Parth (Apache-2.0).", small)]
    doc.build(story)
    return buf.getvalue()
