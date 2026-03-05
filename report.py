"""
report.py — Auto-generate a PDF Q&A session report.
Bonus feature: provides judges a polished export of every question + answer + sources.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict

from config import REPORTS_DIR

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


def generate_qa_report(
    qa_history:    List[Dict],
    output_path:   str = "",
    session_title: str = "HandNotes RAG — Q&A Session Report",
) -> str:
    """
    Generate a formatted PDF report of the Q&A session.

    Args:
        qa_history:   List of result dicts from pipeline.ask().
        output_path:  Where to save. Defaults to REPORTS_DIR/qa_report_<ts>.pdf.
        session_title: Title shown on the cover.

    Returns:
        Absolute path to the generated PDF.
    """
    if not _HAS_REPORTLAB:
        raise ImportError("ReportLab not installed. Run: pip install reportlab")

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(REPORTS_DIR, f"qa_report_{ts}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title=session_title,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    S = lambda name, **kw: ParagraphStyle(name, parent=base["Normal"], **kw)

    title_s    = S("T", fontName="Helvetica-Bold", fontSize=22,
                   textColor=colors.HexColor("#1e293b"), spaceAfter=4)
    subtitle_s = S("ST", fontSize=10,
                   textColor=colors.HexColor("#64748b"), spaceAfter=18)
    q_s        = S("Q", fontName="Helvetica-Bold", fontSize=13,
                   textColor=colors.HexColor("#1e40af"), spaceBefore=14, spaceAfter=5)
    a_s        = S("A", fontSize=11, leading=17,
                   textColor=colors.HexColor("#1e293b"), spaceAfter=6)
    src_s      = S("SRC", fontSize=9, leftIndent=10,
                   textColor=colors.HexColor("#475569"), spaceAfter=3)
    footer_s   = S("FT", fontSize=8, alignment=TA_CENTER,
                   textColor=colors.HexColor("#94a3b8"))

    def _conf_style(conf: float) -> ParagraphStyle:
        if conf >= 0.75: col = "#16a34a"
        elif conf >= 0.45: col = "#d97706"
        else: col = "#dc2626"
        return S(f"C{int(conf*100)}", fontSize=9, fontName="Helvetica-Bold",
                 textColor=colors.HexColor(col))

    # ── Story ─────────────────────────────────────────────────────────────────
    story = []

    # Cover
    story.append(Paragraph(session_title, title_s))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}  ·  "
        f"{len(qa_history)} question(s) answered",
        subtitle_s,
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#e2e8f0"), spaceAfter=8))

    # Summary stats table
    if qa_history:
        grounded = sum(1 for q in qa_history if q.get("is_grounded", False))
        avg_conf = sum(q.get("confidence", 0.0) for q in qa_history) / len(qa_history)
        stats_rows = [
            ["Total Questions", "Answered from Notes", "Avg. Confidence"],
            [
                str(len(qa_history)),
                f"{grounded} ({int(grounded / len(qa_history) * 100)}%)",
                f"{int(avg_conf * 100)}%",
            ],
        ]
        col_w = [5.3 * cm, 5.3 * cm, 5.3 * cm]
        tbl = Table(stats_rows, colWidths=col_w)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING",   (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.HexColor("#475569")),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.6 * cm))

    # Q&A entries
    for i, qa in enumerate(qa_history):
        elems = []

        elems.append(Paragraph(f"Q{i + 1}. {qa.get('question', '')}", q_s))

        answer = qa.get("answer", "").replace("\n", "<br/>")
        elems.append(Paragraph(answer, a_s))

        conf  = qa.get("confidence", 0.0)
        label = qa.get("confidence_label", "")
        if not label:
            if conf >= 0.75:   label = "High Confidence"
            elif conf >= 0.45: label = "Medium Confidence"
            else:              label = "Low / No Info"
        elems.append(Paragraph(f"● {label} ({int(conf * 100)}%)", _conf_style(conf)))

        sources = qa.get("sources", [])
        if sources:
            elems.append(Paragraph("Sources:", src_s))
            seen: set = set()
            for s in sources[:4]:
                key = f"{s['pdf']}_{s['page']}"
                if key not in seen:
                    seen.add(key)
                    rel = int(s.get("score", 0) * 100)
                    elems.append(Paragraph(
                        f"  📄 {s['pdf']}  —  Page {s['page']}  (relevance: {rel}%)",
                        src_s,
                    ))

        elems.append(HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#e2e8f0"), spaceAfter=3))
        story.append(KeepTogether(elems))

    # Footer
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(
        "Generated by HandNotes RAG System  ·  "
        "All answers sourced exclusively from uploaded handwritten notes.",
        footer_s,
    ))

    doc.build(story)
    print(f"✅ Report saved → {output_path}")
    return output_path
