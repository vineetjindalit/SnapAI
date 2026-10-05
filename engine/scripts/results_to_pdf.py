#!/usr/bin/env python3
"""
results_to_pdf.py — Convert BENCHMARK_RESULTS.md → BENCHMARK_RESULTS.pdf

Standalone script so the benchmark itself stays dependency-light
(only OpenCV + NumPy). PDF generation uses ReportLab.

Usage:
    python3 scripts/results_to_pdf.py
"""
from __future__ import annotations
import json, datetime, re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "tests" / "datasets" / "results"
SPEC_FILE = ROOT / "tests" / "datasets" / "clips" / "synthetic" / "_specs.json"

PRIMARY = colors.HexColor("#0A2540")
ACCENT  = colors.HexColor("#635BFF")
LIGHT_BG = colors.HexColor("#F6F9FC")
DARK    = colors.HexColor("#1A1F36")
GRAY    = colors.HexColor("#425466")
GREEN   = colors.HexColor("#00875A")
RED     = colors.HexColor("#DE350B")
AMBER   = colors.HexColor("#E69520")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=24,
                              textColor=PRIMARY, alignment=TA_LEFT,
                              fontName="Helvetica-Bold", spaceAfter=8)
subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11,
                                 textColor=GRAY, spaceAfter=18, leading=16)
h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                           textColor=PRIMARY, fontName="Helvetica-Bold",
                           spaceBefore=16, spaceAfter=8, leading=22,
                           keepWithNext=True)
h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                           textColor=ACCENT, fontName="Helvetica-Bold",
                           spaceBefore=10, spaceAfter=4, leading=16,
                           keepWithNext=True)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                             textColor=DARK, leading=14, spaceAfter=6,
                             alignment=TA_JUSTIFY, fontName="Helvetica")
bullet_style = ParagraphStyle("Bul", parent=body_style, leftIndent=14,
                               bulletIndent=2, spaceAfter=3)
mono_style = ParagraphStyle("Mono", parent=body_style,
                             fontName="Courier", fontSize=9, leading=12)


def load_latest_results() -> tuple[list, dict]:
    """Find the newest benchmark_*.json and parse it."""
    candidates = sorted(RESULTS_DIR.glob("benchmark_*.json"))
    if not candidates:
        raise SystemExit("No benchmark_*.json found — run scripts/benchmark_clips.py first")
    latest = candidates[-1]
    print(f"Reading: {latest.name}")
    data = json.loads(latest.read_text())
    specs = {}
    if SPEC_FILE.exists():
        specs = {s["name"]: s for s in json.loads(SPEC_FILE.read_text())}
    return data, specs


def style_table(t):
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  PRIMARY),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  9),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 8.5),
        ("TEXTCOLOR",   (0,1), (-1,-1), DARK),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]))
    return t


def p(text, style=body_style):
    return Paragraph(text, style)


def add_page_decoration(canv, doc):
    canv.saveState()
    if canv.getPageNumber() > 1:
        canv.setStrokeColor(colors.HexColor("#E5E7EB"))
        canv.line(2*cm, 1.6*cm, A4[0]-2*cm, 1.6*cm)
        canv.setFont("Helvetica", 8)
        canv.setFillColor(GRAY)
        canv.drawString(2*cm, 1.1*cm, "SnapAI v2.5 — Benchmark Results")
        canv.drawRightString(A4[0]-2*cm, 1.1*cm, f"Page {canv.getPageNumber()}")
    canv.restoreState()


def build_pdf(results: list, specs: dict, out_path: Path):
    story = []

    # ── Cover header ────────────────────────────────────────────────
    story.append(Paragraph("SnapAI v2.5", title_style))
    story.append(Paragraph(
        "Clip Dataset Benchmark Results", ParagraphStyle(
            "Sub2", parent=title_style, fontSize=15, textColor=ACCENT,
            spaceAfter=10)))
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        "<b>Pipeline component tested:</b> Priority-tier decision engine "
        "(<i>backend/models/event_engine.py</i>)<br/>"
        "<b>Test corpus:</b> 10 deterministic synthetic clips + optional real clips<br/>"
        "<b>Methodology:</b> Per-frame lightweight signal extraction → "
        "production <code>decide_capture()</code> → priority-floor + cooldown",
        subtitle_style))

    # ── Headline ────────────────────────────────────────────────────
    story.append(Paragraph("Headline", h1_style))
    total_clips = len(results)
    total_caps  = sum(r.get("captures_after_cooldown", 0) for r in results)
    total_frames= sum(r.get("frames_processed", 0) for r in results)
    passes = sum(1 for r in results if r.get("_spec_check", {}).get("pass"))
    checked= sum(1 for r in results if r.get("_spec_check", {}).get("checked"))
    headline_data = [
        ["Metric", "Value"],
        ["Clips processed", str(total_clips)],
        ["Frames analysed", f"{total_frames:,}"],
        ["Total captures fired", str(total_caps)],
        ["Spec assertions passing", f"{passes} / {checked}"],
    ]
    story.append(style_table(Table(headline_data, colWidths=[6*cm, 5*cm])))

    # ── Per-clip table ──────────────────────────────────────────────
    story.append(Paragraph("Per-clip summary", h1_style))
    rows = [["Clip", "Prompt", "Frames", "Captures", "Tiers seen",
             "Avg score", "Max score", "Spec"]]
    for r in results:
        if "error" in r:
            rows.append([r["file"], "—", "—", "ERROR", "—", "—", "—",
                         r["error"][:30]])
            continue
        tiers = ", ".join(f"{k[0]}:{v}"
                          for k, v in r["tier_counts"].items() if v > 0) or "—"
        chk = r.get("_spec_check", {})
        spec_cell = "PASS" if chk.get("pass") else ("FAIL" if chk.get("checked") else "—")
        rows.append([
            r["file"].replace(".mp4", ""),
            r.get("prompt") or "—",
            str(r["frames_processed"]),
            str(r["captures_after_cooldown"]),
            tiers,
            str(r["avg_ensemble_score"]),
            str(r["max_ensemble_score"]),
            spec_cell,
        ])
    table = Table(rows, colWidths=[3.3*cm, 2.4*cm, 1.4*cm, 1.6*cm,
                                    3*cm, 1.7*cm, 1.7*cm, 1.4*cm])
    style_table(table)
    # Color spec column
    for i, r in enumerate(results, start=1):
        chk = r.get("_spec_check", {})
        if chk.get("pass"):
            color = GREEN
        elif chk.get("checked"):
            color = RED
        else:
            color = GRAY
        table.setStyle(TableStyle([
            ("TEXTCOLOR", (7, i), (7, i), color),
            ("FONTNAME", (7, i), (7, i), "Helvetica-Bold"),
        ]))
    story.append(table)
    story.append(Spacer(1, 0.4*cm))

    # ── Methodology note ────────────────────────────────────────────
    story.append(Paragraph("How this benchmark works", h1_style))
    story.append(p(
        "For each clip, every other frame is processed (mimicking the live "
        "capture loop). The harness extracts lightweight signals — motion "
        "(frame-difference), brightness, sharpness (variance of Laplacian), "
        "a skin-tone face proxy, a gaze proxy, a NIMA-style aesthetic proxy, "
        "and a predictor trend — and pipes them into the production "
        "<code>decide_capture()</code> function in "
        "<code>backend/models/event_engine.py</code>."))
    story.append(p(
        "The benchmark also enforces the priority-floor (a 0.4 s minimum gap "
        "between back-to-back priority captures) so a 10-second prompted "
        "moment doesn't fire on every frame. This matches the production "
        "<code>SNAPPY_PRIORITY_FLOOR_SEC</code> behaviour added in v2.5."))
    story.append(p(
        "<b>This benchmark does NOT invoke CLIP, NIMA, or YOLO.</b> Those "
        "models require GPU + multi-GB weights and are out of scope for a "
        "CI-friendly priority-tier benchmark. For full model-stack "
        "validation, run the live server and upload each clip via "
        "<code>/sessions/&lt;id&gt;/upload-video</code>."))

    # ── Per-clip detail ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Per-clip detail", h1_style))
    for r in results:
        story.append(Paragraph(r["file"], h2_style))
        spec = specs.get(r["file"], {})
        if spec:
            story.append(p(f"<b>Scenario:</b> {spec.get('scenario','—')}"))
            story.append(p(f"<b>Expected priority tiers:</b> "
                           f"{', '.join(spec.get('expected_priority', []))} · "
                           f"<b>expected captures:</b> "
                           f"{spec.get('expected_captures_min',0)}–"
                           f"{spec.get('expected_captures_max',999)}"))
            story.append(p(f"<i>{spec.get('notes','')}</i>"))
        if "error" in r:
            story.append(p(f"<b>ERROR:</b> {r['error']}"))
            continue
        chk = r.get("_spec_check", {})
        verdict = ("<font color='#00875A'><b>PASS ✓</b></font>"
                   if chk.get("pass")
                   else "<font color='#DE350B'><b>FAIL ✗</b></font>" if chk.get("checked")
                   else "—")
        story.append(p(
            f"<b>Frames processed:</b> {r['frames_processed']} "
            f"(of {r['frames_total']}) · "
            f"<b>captures:</b> {r['captures_after_cooldown']} "
            f"(triggered pre-cooldown: {r['triggered_count']}) · "
            f"<b>verdict:</b> {verdict}"))
        # Tier breakdown
        tiers = r["tier_counts"]
        story.append(p(
            f"<b>Tier distribution</b> — "
            f"critical: {tiers.get('critical',0)} · "
            f"high: {tiers.get('high',0)} · "
            f"elevated: {tiers.get('elevated',0)} · "
            f"normal: {tiers.get('normal',0)}"))
        story.append(p(
            f"<b>Ensemble score</b> — avg: {r['avg_ensemble_score']}, "
            f"max: {r['max_ensemble_score']}"))
        # First 8 captures as a small table
        if r["captures"]:
            cap_rows = [["Frame", "Tier", "Score", "Thr.",
                         "Class", "Bypass"]]
            for c in r["captures"][:8]:
                cap_rows.append([
                    str(c["frame"]), c["tier"], str(c["score"]),
                    str(c["threshold"]), c["moment_class"],
                    "yes" if c["bypassed_cooldown"] else "—",
                ])
            t = Table(cap_rows, colWidths=[1.6*cm, 2*cm, 1.6*cm, 1.4*cm,
                                            3*cm, 1.8*cm])
            style_table(t)
            story.append(t)
            if len(r["captures"]) > 8:
                story.append(p(f"<i>(showing first 8 of "
                               f"{len(r['captures'])} captures)</i>"))
        story.append(Spacer(1, 0.25*cm))

    # ── Conclusion ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Conclusion", h1_style))
    if checked and passes == checked:
        story.append(p(
            f"<b>All {passes} spec assertions pass.</b> The v2.5 priority-tier "
            "system behaves correctly across the test corpus: prompt-matched "
            "moments fire CRITICAL with cooldown bypass, group-gaze frames "
            "fire HIGH, dark and blank scenes produce zero captures, and "
            "the priority-floor debounce keeps total captures sensible "
            "during long priority moments."))
    elif checked:
        story.append(p(
            f"<b>{passes} of {checked} spec assertions pass.</b> See the "
            "per-clip detail above for which scenarios are still off-spec."))
    story.append(p(
        "Next step: run the same clips through the live model stack "
        "(CLIP + NIMA + YOLOv8-face + HSEmotion) via the server's video "
        "upload endpoint, and compare the capture rate against this "
        "lightweight benchmark. Significant divergence would indicate "
        "the lightweight signal extractors need calibration to match the "
        "full ML stack's behaviour."))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="SnapAI Benchmark Results", author="SnapAI",
    )
    doc.build(story, onFirstPage=add_page_decoration,
              onLaterPages=add_page_decoration)
    print(f"Written PDF: {out_path}")


def main():
    results, specs = load_latest_results()
    out_pdf = RESULTS_DIR / "BENCHMARK_RESULTS.pdf"
    build_pdf(results, specs, out_pdf)


if __name__ == "__main__":
    main()
