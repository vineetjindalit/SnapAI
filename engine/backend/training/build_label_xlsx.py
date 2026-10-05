"""
training/build_label_xlsx.py — turn labels.csv into a visual Excel labelling
sheet: each row shows the FRAME THUMBNAIL embedded next to its predicted label,
a dropdown to correct it, and a yellow flag on low-confidence rows.

Output: Birthday_frames/labels_visual.xlsx

Columns:  image | video | frame_file | t(s) | model_pred | label(dropdown) | confidence
"""
from __future__ import annotations

import csv
import os
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent.parent

# Birthday label taxonomy (snake_case for the model). Order = the user's order;
# general_peak (catch-all candid) + negative (junk/transition) kept at the end.
VOCAB = [
    "pre_preparation", "person_arrival", "surprise_celebration",
    "food_table", "cake", "cake_person", "cake_with_candles", "candle_blowing",
    "cake_cutting", "cake_smashing", "cake_feeding", "birthday_gifting",
    "group_photo", "individual_people",
    "smiling_moments", "laughing_moments", "crying_moments",
    "hugging_moments", "dancing_moments", "gazing_moments",
    "general_peak", "negative",
]
# Human-readable description shown on the Instructions sheet.
VOCAB_DESC = {
    "pre_preparation":     "setting up — decorations, table, balloons (before guests)",
    "person_arrival":      "the birthday person / guests arriving / entrance",
    "surprise_celebration":"the surprise reveal — hands up, shocked faces, 'surprise!'",
    "food_table":          "snacks / food spread on the table (often beside the cake)",
    "cake":                "the cake on its own (close-up of just the cake)",
    "cake_person":         "the birthday person posing WITH the cake",
    "cake_with_candles":   "people at a lit cake, nobody blowing yet",
    "candle_blowing":      "the actual blow — leaning in, cheeks puffed, flames out",
    "cake_cutting":        "knife cutting the cake",
    "cake_smashing":       "face/hands smashing into the cake",
    "cake_feeding":        "feeding a bite of cake to someone",
    "birthday_gifting":    "giving / opening / holding gifts",
    "group_photo":         "a posed group shot",
    "individual_people":   "a solo portrait of one person",
    "smiling_moments":     "a clear smile",
    "laughing_moments":    "open-mouthed laughter",
    "crying_moments":      "tears / emotional crying",
    "hugging_moments":     "a hug / embrace",
    "dancing_moments":     "people dancing",
    "gazing_moments":      "looking into the camera / a held gaze",
    "general_peak":        "a nice candid that doesn't fit a specific class above",
    "negative":            "boring / transitional / no real moment",
}
# Old model classes → new taxonomy (for the pre-filled predictions).
REMAP = {"hug_moment": "hugging_moments", "clapping_scene": "general_peak"}


def _remap(lbl: str) -> str:
    return REMAP.get(lbl, lbl)

THUMB_W, THUMB_H = 200, 150          # bounding box for the embedded thumbnail (px)
LOW_CONF = 0.66                       # rows below this get a yellow flag to review


def _find_manifest() -> Path | None:
    env = os.environ.get("SNAPPY_FRAMES_CSV")
    if env and Path(env).exists():
        return Path(env)
    hits = sorted(ROOT.glob("**/Birthday_frames/labels.csv"))
    return hits[0] if hits else None


def _thumb(path: Path) -> tuple:
    """Return (BytesIO JPEG, w, h) of `path` fitted into THUMB_W×THUMB_H.

    JPEG (not PNG) keeps the workbook small — ~12KB/thumb vs ~60KB for PNG,
    so 2,356 frames fit in ~30MB instead of ~140MB.
    """
    im = PILImage.open(path).convert("RGB")
    im.thumbnail((THUMB_W, THUMB_H))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return buf, im.width, im.height


def _frame_index(frame_file: str) -> int:
    """'frame_0006.jpg' -> 6."""
    try:
        return int(frame_file.split("_")[1].split(".")[0])
    except Exception:
        return -1


def _runs(rows: list) -> list:
    """Collapse consecutive same-prediction frames (per video) into runs.

    rows are in manifest order (by video, then frame index), so a run is a
    maximal block of adjacent frames in one video sharing the same model_pred.
    Returns dicts: video, pred, start_idx, end_idx, rows[].
    """
    runs, cur = [], None
    for r in rows:
        idx = _frame_index(r["frame_file"])
        pred = r.get("model_pred", "") or r.get("label", "")
        if (cur and cur["video"] == r["video"] and cur["pred"] == pred
                and idx == cur["end_idx"] + 1):
            cur["end_idx"] = idx; cur["rows"].append(r)
        else:
            if cur:
                runs.append(cur)
            cur = {"video": r["video"], "pred": pred,
                   "start_idx": idx, "end_idx": idx, "rows": [r]}
    if cur:
        runs.append(cur)
    return runs


def _hdr(ws, headers):
    hfont = Font(name="Arial", bold=True, color="FFFFFF")
    hfill = PatternFill("solid", fgColor="305496")
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = hfont; cell.fill = hfill
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _make_vocab_sheet(wb):
    """Put the (21-item) label list on a hidden sheet and return a range ref.

    The list is too long for Excel's 255-char inline dropdown, so dropdowns
    reference this range instead.
    """
    vs = wb.create_sheet("vocab")
    for i, v in enumerate(VOCAB, start=1):
        vs.cell(row=i, column=1, value=v)
    vs.sheet_state = "hidden"
    return f"vocab!$A$1:$A${len(VOCAB)}"


def main() -> None:
    csv_path = _find_manifest()
    if csv_path is None:
        print("No Birthday_frames/labels.csv found.")
        return
    frames_dir = csv_path.parent
    rows = list(csv.DictReader(open(csv_path)))
    print(f"Building visual sheet for {len(rows)} frames from {csv_path}")

    wb = Workbook()
    buffers = []   # keep all image buffers alive until save

    # ── Sheet 1: Instructions ─────────────────────────────────────────────
    _build_instructions(wb.active)

    # Long label list lives on a hidden sheet; dropdowns reference it.
    vocab_ref = _make_vocab_sheet(wb)

    # ── Sheet 2: RUNS — label continuous blocks in one go ─────────────────
    _build_runs_sheet(wb.create_sheet("LABEL HERE (runs)"),
                      rows, frames_dir, buffers, vocab_ref)

    # ── Sheet 3: per-frame detail (fine-grained fixes) ────────────────────
    ws = wb.create_sheet("frames (detail)")

    headers = ["image", "video", "frame_file", "t(s)", "model_pred",
               "label (edit)", "confidence"]
    _hdr(ws, headers)

    # Column widths (col A sized to the thumbnail; ~px/7)
    ws.column_dimensions["A"].width = THUMB_W / 7.0 + 1
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 7
    ws.column_dimensions["E"].width = 18
    ws.column_dimensions["F"].width = 18
    ws.column_dimensions["G"].width = 11
    ws.freeze_panes = "B2"

    # Dropdown for the editable label column (F) — references the vocab sheet.
    dv = DataValidation(type="list", formula1=vocab_ref, allow_blank=True)
    ws.add_data_validation(dv)

    afont = Font(name="Arial")
    flag  = PatternFill("solid", fgColor="FFF2CC")     # light yellow
    center = Alignment(horizontal="center", vertical="center")

    for i, r in enumerate(rows, start=2):
        fp = frames_dir / r["video"] / r["frame_file"]
        try:
            buf, w, h = _thumb(fp)
            buffers.append(buf)
            img = XLImage(buf); img.width = w; img.height = h
            ws.add_image(img, f"A{i}")
        except Exception:
            ws.cell(row=i, column=1, value="(missing)")
        ws.row_dimensions[i].height = THUMB_H * 0.75      # px→points

        ws.cell(row=i, column=2, value=r["video"]).font = afont
        ws.cell(row=i, column=3, value=r["frame_file"]).font = afont
        ws.cell(row=i, column=4, value=r.get("t_seconds", "")).font = afont
        ws.cell(row=i, column=5, value=r.get("model_pred", "")).font = afont
        lab = ws.cell(row=i, column=6, value=_remap(r.get("label", "")))
        lab.font = Font(name="Arial", bold=True); lab.alignment = center
        dv.add(lab)
        conf_cell = ws.cell(row=i, column=7, value=float(r["confidence"])
                            if r.get("confidence") else None)
        conf_cell.font = afont; conf_cell.alignment = center
        conf_cell.number_format = "0.000"
        # Flag low-confidence rows for priority review
        try:
            if r.get("confidence") and float(r["confidence"]) < LOW_CONF:
                lab.fill = flag
        except ValueError:
            pass
        for col in range(2, 8):
            ws.cell(row=i, column=col).alignment = (
                center if col in (4, 5, 6, 7)
                else Alignment(vertical="center"))

    wb.active = wb.sheetnames.index("LABEL HERE (runs)")   # open on the runs sheet
    out = frames_dir / "labels_visual.xlsx"
    wb.save(out)
    size_mb = out.stat().st_size / 1e6
    print(f"Saved {out}  ({size_mb:.1f} MB, {len(rows)} frames, "
          f"{len(_runs(rows))} runs)")


def _build_runs_sheet(ws, rows, frames_dir, buffers, vocab_ref) -> None:
    """One row per CONTINUOUS run of same-prediction frames.

    Editing the single 'label (edit)' dropdown here relabels the WHOLE block
    of frames at once — the fast path for continuous labelling. A thumbnail of
    the run's middle frame gives visual context.
    """
    runs = _runs(rows)
    _hdr(ws, ["image", "video", "frames", "count", "t range",
              "model_pred", "label (edit) — sets whole run"])
    ws.column_dimensions["A"].width = THUMB_W / 7.0 + 1
    for col, w in zip("BCDEFG", (34, 12, 7, 14, 18, 30)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B2"

    dv = DataValidation(type="list", formula1=vocab_ref, allow_blank=True)
    ws.add_data_validation(dv)
    afont = Font(name="Arial"); center = Alignment("center", vertical="center")
    flag = PatternFill("solid", fgColor="FFF2CC")

    for i, run in enumerate(runs, start=2):
        mid = run["rows"][len(run["rows"]) // 2]
        fp = frames_dir / run["video"] / mid["frame_file"]
        try:
            buf, w, h = _thumb(fp); buffers.append(buf)
            img = XLImage(buf); img.width = w; img.height = h
            ws.add_image(img, f"A{i}")
        except Exception:
            ws.cell(row=i, column=1, value="(missing)")
        ws.row_dimensions[i].height = THUMB_H * 0.75
        t0 = run["rows"][0].get("t_seconds", ""); t1 = run["rows"][-1].get("t_seconds", "")
        ws.cell(row=i, column=2, value=run["video"]).font = afont
        ws.cell(row=i, column=3,
                value=f"{run['start_idx']}–{run['end_idx']}").font = afont
        ws.cell(row=i, column=4, value=len(run["rows"])).font = afont
        ws.cell(row=i, column=5, value=f"{t0}–{t1}s").font = afont
        ws.cell(row=i, column=6, value=run["pred"]).font = afont
        lab = ws.cell(row=i, column=7, value=_remap(run["pred"]))
        lab.font = Font(name="Arial", bold=True); lab.alignment = center
        dv.add(lab)
        if len(run["rows"]) == 1:        # singletons are often the uncertain ones
            lab.fill = flag
        for col in (3, 4, 5, 6, 7):
            ws.cell(row=i, column=col).alignment = center


def _build_instructions(ws) -> None:
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 100
    lines = [
        ("SnapAI — Birthday frame labelling", True),
        ("", False),
        ("You have TWO ways to label. Use whichever is faster:", True),
        ("", False),
        ("① 'LABEL HERE (runs)' sheet  ← RECOMMENDED, fastest", True),
        ("   • Each row = a CONTINUOUS block of frames the model gave the same label.", False),
        ("   • Change the dropdown in the last column → it relabels the WHOLE block at once.", False),
        ("   • The thumbnail shows the middle frame of that block for context.", False),
        ("   • Yellow rows are single-frame runs (often the uncertain ones).", False),
        ("", False),
        ("② 'frames (detail)' sheet  ← for fine-grained fixes", True),
        ("   • One row per frame, with its thumbnail + dropdown.", False),
        ("   • Use it when a block needs splitting (e.g. the blow is only 2 frames of a run).", False),
        ("", False),
        ("BULK-FILL TRICK (works in Excel & LibreOffice):", True),
        ("   • Select several cells in the label column (click first, Shift-click last).", False),
        ("   • Type the label (or pick from the dropdown on the active cell).", False),
        ("   • Press  Ctrl+Enter  → fills ALL selected cells with that label at once.", False),
        ("   • Or set one cell, then drag its bottom-right corner down to copy.", False),
        ("", False),
        ("Labels to use (pick from the dropdown):", True),
    ]
    lines += [(f"   {k:20} — {VOCAB_DESC[k]}", False) for k in VOCAB]
    lines += [
        ("", False),
        ("When done: save as .xlsx and tell the assistant — it reads your labels", True),
        ("and rebuilds the model from them. Editing the runs sheet is enough;", False),
        ("you don't have to touch every frame.", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name="Arial", bold=bold,
                      size=13 if (bold and i == 1) else 11)


if __name__ == "__main__":
    main()
