#!/usr/bin/env python3
"""Convert report/REPORT.md to report/REPORT.docx with embedded figures.

A small Markdown subset is supported: headings, paragraphs, bullet/numbered
lists, fenced code blocks, blockquotes, pipe tables and images.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = REPO_ROOT / "report" / "REPORT.md"
REPORT_DOCX = REPO_ROOT / "report" / "REPORT.docx"
MAX_IMAGE_WIDTH_IN = 6.3
MAX_IMAGE_HEIGHT_IN = 6.9


def add_inline_runs(paragraph, text: str) -> None:
    """Add markdown bold/italic/code/link spans as runs (Satır içi biçimlendirme)."""
    token_pattern = re.compile(r"(\*\*.+?\*\*|`.+?`|\[[^\]]+\]\([^)]+\)|\*[^*]+\*)")
    pos = 0
    for match in token_pattern.finditer(text):
        if match.start() > pos:
            paragraph.add_run(text[pos:match.start()])
        token = match.group(0)
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x8B, 0x1A, 0x1A)
        elif token.startswith("[") and "](" in token:
            label = token[1:token.index("]")]
            paragraph.add_run(label)
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        pos = match.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())) and "-" in line


def parse_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def add_table(doc: Document, rows: list[list[str]]) -> None:
    """Add a pipe table to the document (Markdown tablosunu Word tablosuna çevir)."""
    n_cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    if n_cols >= 8:
        widths = [0.95, 0.58, 0.65, 0.48, 0.48, 0.48, 0.50, 0.48, 0.48, 0.48]
        widths = widths[:n_cols]
        font_size = 7.5
    elif n_cols == 7 and rows[0][0] == "Fotoğraf":
        widths = [1.05, 0.62, 0.85, 0.85, 0.85, 0.65, 0.55]
        font_size = 8.5
    elif n_cols == 7:
        widths = [1.25, 0.70, 0.70, 0.65, 0.85, 0.65, 0.90]
        font_size = 8.5
    elif n_cols == 2:
        widths = [2.2, 4.1]
        font_size = 9
    else:
        widths = [6.3 / n_cols] * n_cols
        font_size = 9
    for c_idx, width in enumerate(widths):
        table.columns[c_idx].width = Inches(width)
    for r_idx, row in enumerate(rows):
        tr_pr = table.rows[r_idx]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if r_idx == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for c_idx in range(n_cols):
            cell = table.cell(r_idx, c_idx)
            cell.width = Inches(widths[c_idx])
            cell.text = ""
            paragraph = cell.paragraphs[0]
            text = row[c_idx] if c_idx < len(row) else ""
            add_inline_runs(paragraph, text)
            for run in paragraph.runs:
                run.font.size = Pt(font_size)
                if r_idx == 0:
                    run.bold = True
    doc.add_paragraph()


def add_image(doc: Document, alt: str, rel_path: str) -> None:
    """Insert an image scaled to the page width (Görseli sayfa genişliğine sığdır)."""
    image_path = (REPORT_MD.parent / rel_path).resolve()
    if not image_path.exists():
        doc.add_paragraph(f"[Eksik görsel: {rel_path}]")
        return
    with Image.open(image_path) as img:
        width, height = img.size
    width_in = MAX_IMAGE_WIDTH_IN
    height_in = width_in * height / width
    if height_in > MAX_IMAGE_HEIGHT_IN:  # Keep images on one page (sayfaya sığdır)
        height_in = MAX_IMAGE_HEIGHT_IN
        width_in = height_in * width / height
    if doc.paragraphs and doc.paragraphs[-1].text.strip().startswith("foto"):
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
    doc.add_picture(str(image_path), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if alt:
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run(alt)
        run.italic = True
        run.font.size = Pt(9)


def convert() -> None:
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    in_code = False
    table_buffer: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_buffer
        if table_buffer:
            add_table(doc, table_buffer)
            table_buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            in_code = not in_code
            i += 1
            continue
        if in_code:
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            if not is_table_separator(stripped):
                table_buffer.append(parse_table_row(stripped))
            i += 1
            continue
        flush_table()

        if not stripped:
            i += 1
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            add_image(doc, image_match.group(1), image_match.group(2))
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            if level == 1:
                paragraph = doc.add_heading(level=0)
            else:
                paragraph = doc.add_heading(level=min(level - 1, 9))
            add_inline_runs(paragraph, text)
            i += 1
            continue

        if stripped.startswith(">"):
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(stripped.lstrip("> ").strip())
            run.italic = True
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            i += 1
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            paragraph = doc.add_paragraph(style="List Bullet")
            add_inline_runs(paragraph, bullet_match.group(1))
            i += 1
            continue

        number_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if number_match:
            paragraph = doc.add_paragraph(style="List Number")
            add_inline_runs(paragraph, number_match.group(1))
            i += 1
            continue

        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            doc.add_paragraph("_" * 60)
            i += 1
            continue

        paragraph = doc.add_paragraph()
        add_inline_runs(paragraph, stripped)
        i += 1

    flush_table()
    doc.save(REPORT_DOCX)
    print(f"Saved {REPORT_DOCX}")


if __name__ == "__main__":
    convert()
