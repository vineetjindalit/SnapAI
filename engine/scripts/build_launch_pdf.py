"""
scripts/build_launch_pdf.py — render LAUNCH_ROADMAP.md → printable HTML.

Why HTML and not real PDF?
  Generating a true PDF needs reportlab/weasyprint/etc. which add a
  heavy dependency for a one-off doc. HTML → "Print → Save as PDF" in
  Chrome/Safari produces a perfect PDF in 5 seconds.

Run:
    python3 scripts/build_launch_pdf.py
    open snappy_final/LAUNCH_ROADMAP.html
    Cmd+P → "Save as PDF"
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Multiple docs renderable: LAUNCH_ROADMAP and OWNERSHIP. Pass --doc=name.md
SRC  = ROOT / "LAUNCH_ROADMAP.md"
OUT  = ROOT / "LAUNCH_ROADMAP.html"


_CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #18181b; max-width: 900px; margin: 0 auto;
  padding: 28px 32px;
}
h1 { font-size: 24pt; color: #1e1b4b; margin: 0 0 4px 0;
     border-bottom: 3px solid #6366f1; padding-bottom: 6px; }
h2 { font-size: 17pt; color: #1e1b4b; margin: 28px 0 10px 0;
     padding-top: 18px; border-top: 1px solid #e5e7eb; }
h3 { font-size: 13pt; color: #4338ca; margin: 18px 0 6px 0; }
h4 { font-size: 11pt; color: #1e293b; margin: 12px 0 4px 0; }
p, li { margin: 4px 0; }
blockquote { border-left: 3px solid #6366f1; padding: 6px 14px;
  background: #eef2ff; margin: 12px 0; color: #1e1b4b; font-size: 10pt; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9.5pt;
  background: #f1f5f9; padding: 1px 5px; border-radius: 3px; color: #0f172a; }
pre { background: #0f172a; color: #f1f5f9; padding: 10px 12px;
  border-radius: 6px; overflow-x: auto; font-size: 9pt; line-height: 1.45; }
pre code { background: transparent; color: inherit; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
  font-size: 9.5pt; page-break-inside: avoid; }
th { background: #1e1b4b; color: #ffffff; font-weight: 600;
  padding: 6px 8px; text-align: left; border: 1px solid #312e81; }
td { padding: 5px 8px; border: 1px solid #e5e7eb; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
ul, ol { padding-left: 22px; margin: 6px 0; }
hr { border: none; border-top: 1px dashed #cbd5e1; margin: 20px 0; }
.cover { text-align: center; padding: 80px 0 40px 0; page-break-after: always; }
.cover h1 { border: none; font-size: 38pt; }
.cover .sub { font-size: 14pt; color: #475569; margin-top: 4px; }
.cover .meta { font-size: 10pt; color: #64748b; margin-top: 36px; }
.toc { page-break-after: always; }
.toc h2 { border: none; padding-top: 0; }
.toc ol { font-size: 11pt; line-height: 2; }
.tag-ok   { color: #15803d; font-weight: 600; }
.tag-warn { color: #b45309; font-weight: 600; }
.tag-bad  { color: #b91c1c; font-weight: 600; }
@media print {
  body { padding: 0; }
  h2 { page-break-before: auto; }
  h2, h3, h4 { page-break-after: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
}
"""


def _md_to_html(md: str) -> str:
    """Tiny markdown→HTML — handles only the subset we use in this doc."""
    out_lines = []
    in_code = False
    in_table = False
    table_buf = []
    in_list = None   # "ul" | "ol" | None
    in_blockquote = False

    def flush_table():
        nonlocal in_table, table_buf
        if not in_table:
            return
        rows = [r for r in table_buf if r.strip()]
        if len(rows) >= 2:
            head = [c.strip() for c in rows[0].strip("|").split("|")]
            body_rows = rows[2:]   # skip the |---|---| separator
            out_lines.append("<table>")
            out_lines.append("<thead><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in head) + "</tr></thead>")
            out_lines.append("<tbody>")
            for r in body_rows:
                cells = [c.strip() for c in r.strip("|").split("|")]
                out_lines.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            out_lines.append("</tbody></table>")
        in_table = False
        table_buf = []

    def close_list():
        nonlocal in_list
        if in_list:
            out_lines.append(f"</{in_list}>")
            in_list = None

    def close_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            out_lines.append("</blockquote>")
            in_blockquote = False

    for line in md.splitlines():
        if line.strip().startswith("```"):
            flush_table(); close_list(); close_blockquote()
            if not in_code:
                lang = line.strip()[3:]
                out_lines.append(f"<pre><code>")
                in_code = True
            else:
                out_lines.append("</code></pre>")
                in_code = False
            continue
        if in_code:
            out_lines.append(_escape(line))
            continue

        if line.lstrip().startswith("|") and "|" in line[1:]:
            close_list(); close_blockquote()
            in_table = True
            table_buf.append(line)
            continue
        flush_table()

        if line.strip().startswith(">"):
            close_list()
            if not in_blockquote:
                out_lines.append("<blockquote>")
                in_blockquote = True
            out_lines.append(_inline(line.strip().lstrip(">").strip()) + "<br/>")
            continue
        close_blockquote()

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_list()
            level = len(m.group(1))
            out_lines.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            kind = "ol" if m.group(2)[0].isdigit() else "ul"
            if in_list != kind:
                close_list()
                out_lines.append(f"<{kind}>")
                in_list = kind
            out_lines.append(f"<li>{_inline(m.group(3))}</li>")
            continue
        close_list()

        if line.strip() == "---":
            out_lines.append("<hr/>")
            continue

        if not line.strip():
            out_lines.append("")
            continue

        out_lines.append(f"<p>{_inline(line)}</p>")

    flush_table(); close_list(); close_blockquote()
    return "\n".join(out_lines)


def _inline(s: str) -> str:
    s = _escape(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`",       r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = s.replace("✅", '<span class="tag-ok">✅</span>')
    s = s.replace("⚠️", '<span class="tag-warn">⚠️</span>')
    s = s.replace("❌", '<span class="tag-bad">❌</span>')
    return s


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _try_render_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Try Chrome headless first (perfect rendering), then weasyprint, then
    reportlab. Return True on success."""
    import shutil, subprocess, sys

    # 1. Chrome headless on macOS
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists():
        chrome = shutil.which("google-chrome") or shutil.which("chromium") \
              or shutil.which("chromium-browser")
    if chrome:
        try:
            subprocess.run([
                chrome, "--headless", "--disable-gpu", "--no-sandbox",
                f"--print-to-pdf={pdf_path}",
                "--no-pdf-header-footer",
                f"file://{html_path.resolve()}",
            ], check=True, capture_output=True, timeout=60)
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                print(f"✅ rendered via Chrome headless")
                return True
        except Exception as e:
            print(f"   Chrome headless failed: {e}")

    # 2. weasyprint (if installed)
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"✅ rendered via weasyprint")
        return True
    except Exception:
        pass

    # 3. wkhtmltopdf
    wk = shutil.which("wkhtmltopdf")
    if wk:
        try:
            subprocess.run([wk, "--quiet", str(html_path), str(pdf_path)],
                           check=True, capture_output=True, timeout=60)
            print(f"✅ rendered via wkhtmltopdf")
            return True
        except Exception:
            pass

    return False


def main():
    import sys
    docs = []
    if len(sys.argv) > 1:
        docs = [(ROOT / a).resolve() for a in sys.argv[1:]]
    else:
        docs = [ROOT / "LAUNCH_ROADMAP.md", ROOT / "OWNERSHIP.md"]

    for src in docs:
        if not src.exists():
            print(f"⚠️  skip: {src.name} not found"); continue
        out_html = src.with_suffix(".html")
        out_pdf  = src.with_suffix(".pdf")
        _render_one(src, out_html, out_pdf)


def _render_one(src: Path, out_html: Path, out_pdf: Path):
    md = src.read_text()
    body = _md_to_html(md)
    title = src.stem.replace("_", " ").title()
    sub_titles = {
        "LAUNCH_ROADMAP": "Launch Roadmap — v2.5 → v3.0 Production",
        "OWNERSHIP":      "Sprint Ownership — who does what",
    }
    sub = sub_titles.get(src.stem, title)
    cover = f"""
    <section class="cover">
      <h1>SNAPPY</h1>
      <div class="sub">{sub}</div>
      <div class="meta">Engineering plan · Cost model · Pre-launch checklist</div>
    </section>"""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Snappy — {title}</title>
<style>{_CSS}</style></head><body>
{cover}
{body}
</body></html>"""
    out_html.write_text(html)
    print(f"✅ {src.name} → {out_html.name}")
    if _try_render_pdf(out_html, out_pdf):
        print(f"✅ {src.name} → {out_pdf.name} ({out_pdf.stat().st_size/1024:.0f} KB)")
    else:
        print(f"⚠️  no PDF renderer available — open {out_html.name} and Print → Save as PDF")


if __name__ == "__main__":
    main()
