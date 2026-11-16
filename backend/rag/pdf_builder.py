"""PDF builder: HTML -> one-page US Letter PDF with a headless Chromium browser (Chrome, Edge, Chromium).

Holds the resume template CSS, the spacing LEVELS the fit loop walks through, printing, and page-fill reading.
Change the look of the resume here.
"""
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pypdf import PdfReader

import shutil

_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def find_browser() -> Path | None:
    """Any Chromium-based browser can print to PDF headlessly. RAG_BROWSER wins if set."""
    if os.environ.get("RAG_BROWSER"):
        return Path(os.environ["RAG_BROWSER"])
    for c in _CANDIDATES:
        if Path(c).exists():
            return Path(c)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge"):
        if found := shutil.which(name):
            return Path(found)
    return None


EDGE = find_browser()

LEVELS = [
    dict(pt=.36, pb=.26, lh=1.28, li=1.5, ulb=5, h2t=7),
    dict(pt=.34, pb=.24, lh=1.26, li=1.2, ulb=4, h2t=6),
    dict(pt=.32, pb=.22, lh=1.24, li=1.0, ulb=4, h2t=5),
    dict(pt=.30, pb=.20, lh=1.22, li=1.0, ulb=3, h2t=5),
    dict(pt=.28, pb=.18, lh=1.20, li=0.5, ulb=3, h2t=4),
    dict(pt=.24, pb=.14, lh=1.18, li=0.5, ulb=2, h2t=3),
    dict(pt=.20, pb=.10, lh=1.16, li=0.0, ulb=2, h2t=3),
]
DEFAULT_LEVEL = 4

CSS = """
@page {{ size: Letter; margin: 0; }}
body {{ font-family: Cambria, Georgia, 'Times New Roman', serif; color: #111; margin: 0;
  padding: {pt}in 0.52in {pb}in 0.52in; font-size: 9.3pt; line-height: {lh};
  font-variant-numeric: lining-nums tabular-nums; }}
h1 {{ text-align: center; font-size: 19pt; margin: 0 0 2px 0; font-weight: normal; }}
.contact {{ text-align: center; font-size: 8.8pt; margin-bottom: 5px; }}
.contact a {{ color: #111; text-decoration: none; }}
h2 {{ font-variant: small-caps; font-size: 10.7pt; letter-spacing: 0.5px;
  border-bottom: 1px solid #111; margin: {h2t}px 0 2px 0; padding-bottom: 1px; font-weight: bold; }}
.row {{ display: flex; justify-content: space-between; align-items: baseline; }}
.row .left {{ font-weight: bold; }}
.sub-row {{ display: flex; justify-content: space-between; align-items: baseline; font-style: italic; margin-bottom: 1px; }}
ul {{ margin: 1px 0 {ulb}px 0; padding-left: 15px; }}
li {{ text-align: left; margin-bottom: {li}px; }}
p {{ margin: 0 0 2px 0; text-align: left; }}
a.inline {{ color: #111; text-decoration: none; }}
.proj-tech {{ font-weight: normal; font-style: italic; font-size: 8.8pt; text-align: left; margin: 0 0 1px 0; }}
.context {{ margin: 0 0 1px 0; }}
.skills div {{ margin-bottom: 1px; }}
"""


def edge(args):
    if EDGE is None:
        raise RuntimeError("No Chrome, Edge or Chromium found for PDF printing. Install one or set RAG_BROWSER.")
    with tempfile.TemporaryDirectory() as prof:
        return subprocess.run([str(EDGE), "--headless", "--disable-gpu", f"--user-data-dir={prof}",
                               "--no-first-run", "--no-pdf-header-footer", *args],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)


def read_pdf(pdf_path):
    """Read into memory so no file handle stays open (Windows locks open files)."""
    return PdfReader(io.BytesIO(pdf_path.read_bytes()))


def print_pdf(html_path, pdf_path):
    """Print with Edge to a fresh path. Returns the page count."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    edge([f"--print-to-pdf={pdf_path}", html_path.as_uri()])
    for _ in range(20):  # Edge can finish writing a moment after it exits
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("Edge did not produce a PDF.")
    return len(read_pdf(pdf_path).pages)


def page_fill(pdf_path):
    """Share of the page height used, from the lowest line of text on page 1."""
    page = read_pdf(pdf_path).pages[0]
    height = float(page.mediabox.height)
    ys = []

    def visit(text, cm, tm, font, size):
        if text.strip():
            ys.append(tm[4] * cm[1] + tm[5] * cm[3] + cm[5])

    page.extract_text(visitor_text=visit)
    ys = [y for y in ys if 0 <= y <= height]
    return (height - min(ys)) / height if ys else None
