"""
nbsi/ingestion/document_reader.py

Structure-aware document readers.

The problem with basic reading:
    pdfminer extract_text() returns a flat string — columns merge, headers
    run into body text, footnotes appear mid-paragraph. The result fed to
    spaCy as one block produces garbled nodes.

    python-docx paragraph join loses all heading hierarchy — every paragraph
    becomes equal weight, so "Introduction" and "References" look the same
    as body text to the extractor.

What these readers do differently:
    - Preserve heading hierarchy as tagged sections
    - Filter out headers, footers, page numbers, footnotes
    - Detect and skip tables (they produce terrible nodes)
    - Return a DocumentStructure with both raw text and a list of sections
      that can be fed directly to the chunker

Supported:
    PDF    — via pdfminer.six (structure-aware via layout analysis)
    DOCX   — via python-docx (heading style detection)
    HTML   — via stdlib html.parser (tag-based structure)
    MD     — heading detection without external deps
    TXT    — plain text (passes through to chunker)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A single section of a document with its heading and body."""
    heading:    str
    body:       str
    level:      int   = 0    # 0 = no heading, 1 = H1, 2 = H2, etc.
    page:       int   = 0    # page number (PDF only, 0 if unknown)

    def to_text(self) -> str:
        """Return heading + body as a single string for extraction."""
        if self.heading:
            return f"{self.heading}\n\n{self.body}"
        return self.body

    def is_empty(self) -> bool:
        return len(self.body.strip()) < 10


@dataclass
class DocumentStructure:
    """
    The parsed structure of a document.

    sections  : ordered list of Section objects
    title     : document title (from metadata or first H1)
    author    : author string (from metadata, or '')
    date      : date string (from metadata, or '')
    format    : 'pdf' | 'docx' | 'html' | 'md' | 'txt'
    raw_text  : full text as a single string (fallback)
    """
    sections:  list[Section]
    title:     str   = ''
    author:    str   = ''
    date:      str   = ''
    format:    str   = 'txt'
    raw_text:  str   = ''

    def to_plain_text(self) -> str:
        """All sections joined as plain text."""
        if self.sections:
            return '\n\n'.join(s.to_text() for s in self.sections
                               if not s.is_empty())
        return self.raw_text

    def headings(self) -> list[str]:
        return [s.heading for s in self.sections if s.heading]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_document(path: str) -> Optional[DocumentStructure]:
    """
    Read a document and return its structure.
    Returns None if the file cannot be read or format is unsupported.
    Never raises.
    """
    if not os.path.isfile(path):
        return None

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == '.pdf':
            return _read_pdf(path)
        elif ext == '.docx':
            return _read_docx(path)
        elif ext in ('.html', '.htm'):
            return _read_html(path)
        elif ext == '.md':
            return _read_markdown(path)
        elif ext in ('.txt', '.py', '.js', '.ts', '.rst', '.csv'):
            return _read_plain(path)
        else:
            return None
    except Exception as e:
        print(f'[reader] Failed to read {path}: {e}')
        return None


# ---------------------------------------------------------------------------
# PDF reader — structure-aware
# ---------------------------------------------------------------------------

def _read_pdf(path: str) -> Optional[DocumentStructure]:
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import (
            LTPage, LTTextBox, LTTextLine, LTChar, LTAnon,
            LTFigure, LTLayoutContainer
        )
    except ImportError:
        print('[reader] pdfminer.six not installed — falling back to basic read')
        return _read_pdf_basic(path)

    sections   = []
    current_heading = ''
    current_body    = []
    current_level   = 0
    page_num        = 0

    # Font-size threshold: text significantly larger than body = heading
    all_sizes = []

    # First pass: collect all font sizes to establish baseline
    try:
        for page_layout in extract_pages(path):
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    for line in element:
                        if isinstance(line, LTTextLine):
                            for char in line:
                                if isinstance(char, LTChar):
                                    all_sizes.append(char.size)
    except Exception:
        return _read_pdf_basic(path)

    if not all_sizes:
        return _read_pdf_basic(path)

    # Body font is the most common size
    from statistics import mode, mean
    try:
        body_size = mode(all_sizes)
    except Exception:
        body_size = mean(all_sizes)

    heading_threshold = body_size * 1.15   # 15% larger = likely heading

    # Second pass: extract with structure
    try:
        for page_layout in extract_pages(path):
            page_num += 1
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue

                lines      = []
                avg_size   = 0.0
                char_count = 0

                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue
                    line_text = line.get_text().strip()
                    if not line_text:
                        continue
                    lines.append(line_text)
                    for char in line:
                        if isinstance(char, LTChar):
                            avg_size   += char.size
                            char_count += 1

                if not lines:
                    continue

                block_text = ' '.join(lines)
                avg_size   = (avg_size / char_count) if char_count else body_size

                # Skip page numbers (short, numeric)
                if re.match(r'^\d+$', block_text.strip()):
                    continue

                # Skip very short non-heading text (likely headers/footers)
                if len(block_text) < 20 and avg_size <= heading_threshold:
                    continue

                if avg_size >= heading_threshold:
                    # Seal previous section
                    body = '\n'.join(current_body).strip()
                    if body:
                        sections.append(Section(
                            heading=current_heading,
                            body=body,
                            level=current_level,
                            page=page_num,
                        ))
                    current_heading = block_text
                    current_level   = 1 if avg_size >= body_size * 1.3 else 2
                    current_body    = []
                else:
                    current_body.append(block_text)

    except Exception:
        return _read_pdf_basic(path)

    # Seal final section
    body = '\n'.join(current_body).strip()
    if body:
        sections.append(Section(
            heading=current_heading,
            body=body,
            level=current_level,
        ))

    if not sections:
        return _read_pdf_basic(path)

    raw = '\n\n'.join(s.to_text() for s in sections)
    return DocumentStructure(
        sections=sections,
        format='pdf',
        raw_text=raw,
    )


def _read_pdf_basic(path: str) -> Optional[DocumentStructure]:
    """Fallback: flat pdfminer text extraction."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path)
        if not text or not text.strip():
            return None
        return DocumentStructure(
            sections=[Section(heading='', body=text.strip())],
            format='pdf',
            raw_text=text.strip(),
        )
    except Exception as e:
        print(f'[reader] PDF basic read failed: {e}')
        return None


# ---------------------------------------------------------------------------
# DOCX reader — heading-style aware
# ---------------------------------------------------------------------------

def _read_docx(path: str) -> Optional[DocumentStructure]:
    try:
        from docx import Document
    except ImportError:
        print('[reader] python-docx not installed')
        return None

    doc      = Document(path)
    sections = []
    current_heading = ''
    current_level   = 0
    current_body    = []

    # Extract metadata
    core = doc.core_properties
    title  = core.title  or ''
    author = core.author or ''
    date   = str(core.created) if core.created else ''

    for para in doc.paragraphs:
        text  = para.text.strip()
        style = para.style.name if para.style else ''

        if not text:
            continue

        # Detect heading styles: "Heading 1", "Heading 2", "Title", etc.
        heading_match = re.match(r'Heading (\d+)|Title', style)
        if heading_match:
            # Seal previous section
            body = '\n'.join(current_body).strip()
            if body:
                sections.append(Section(
                    heading=current_heading,
                    body=body,
                    level=current_level,
                ))
            level_str = heading_match.group(1)
            current_heading = text
            current_level   = int(level_str) if level_str else 0
            current_body    = []
        else:
            current_body.append(text)

    # Seal final section
    body = '\n'.join(current_body).strip()
    if body:
        sections.append(Section(
            heading=current_heading,
            body=body,
            level=current_level,
        ))

    if not sections:
        # Document has no heading styles — return flat
        all_text = '\n'.join(
            p.text for p in doc.paragraphs if p.text.strip()
        )
        if not all_text:
            return None
        sections = [Section(heading='', body=all_text)]

    raw = '\n\n'.join(s.to_text() for s in sections)
    return DocumentStructure(
        sections=sections,
        title=title,
        author=author,
        date=date,
        format='docx',
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# HTML reader
# ---------------------------------------------------------------------------

def _read_html(path: str) -> Optional[DocumentStructure]:
    from html.parser import HTMLParser

    class _StructureParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sections        = []
            self._current_tag    = ''
            self._current_text   = []
            self._current_level  = 0
            self._current_heading = ''
            self._body_lines     = []
            self._in_skip        = False   # inside script/style/nav
            self._skip_tags      = {'script', 'style', 'nav', 'footer',
                                    'header', 'aside'}
            self._skip_depth     = 0
            self.title           = ''
            self._in_title       = False

        def handle_starttag(self, tag, attrs):
            if tag in self._skip_tags:
                self._in_skip   = True
                self._skip_depth += 1
                return
            if self._in_skip:
                self._skip_depth += 1
                return

            if tag == 'title':
                self._in_title = True

            m = re.match(r'h([1-4])', tag)
            if m:
                # Seal current section
                body = ' '.join(self._body_lines).strip()
                if body:
                    self.sections.append(Section(
                        heading=self._current_heading,
                        body=body,
                        level=self._current_level,
                    ))
                self._current_level   = int(m.group(1))
                self._current_heading = ''
                self._body_lines      = []
                self._current_tag     = tag

        def handle_endtag(self, tag):
            if tag in self._skip_tags and self._in_skip:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._in_skip    = False
                    self._skip_depth = 0
                return
            if self._in_skip:
                self._skip_depth -= 1
                return
            if tag == 'title':
                self._in_title = False
            m = re.match(r'h([1-4])', tag)
            if m:
                self._current_heading = ' '.join(self._current_text).strip()
                self._current_text    = []
                self._current_tag     = ''

        def handle_data(self, data):
            if self._in_skip:
                return
            text = data.strip()
            if not text:
                return
            if self._in_title:
                self.title += text
                return
            if self._current_tag and re.match(r'h[1-4]', self._current_tag):
                self._current_text.append(text)
            else:
                self._body_lines.append(text)

        def finalise(self):
            body = ' '.join(self._body_lines).strip()
            if body:
                self.sections.append(Section(
                    heading=self._current_heading,
                    body=body,
                    level=self._current_level,
                ))

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()

    parser = _StructureParser()
    parser.feed(html)
    parser.finalise()

    if not parser.sections:
        # Strip all tags and return plain
        plain = re.sub(r'<[^>]+>', ' ', html)
        plain = re.sub(r'\s+', ' ', plain).strip()
        if not plain:
            return None
        return DocumentStructure(
            sections=[Section(heading='', body=plain)],
            title=parser.title,
            format='html',
            raw_text=plain,
        )

    raw = '\n\n'.join(s.to_text() for s in parser.sections)
    return DocumentStructure(
        sections=parser.sections,
        title=parser.title,
        format='html',
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# Markdown reader
# ---------------------------------------------------------------------------

def _read_markdown(path: str) -> Optional[DocumentStructure]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    if not text.strip():
        return None

    sections = []
    current_heading = ''
    current_level   = 0
    current_lines   = []
    title           = ''

    for line in text.splitlines():
        m = re.match(r'^(#{1,4})\s+(.+)', line)
        if m:
            # Seal current section
            body = '\n'.join(current_lines).strip()
            if body:
                sections.append(Section(
                    heading=current_heading,
                    body=body,
                    level=current_level,
                ))
            level           = len(m.group(1))
            current_heading = m.group(2).strip()
            current_level   = level
            current_lines   = []
            if level == 1 and not title:
                title = current_heading
        else:
            current_lines.append(line)

    body = '\n'.join(current_lines).strip()
    if body:
        sections.append(Section(
            heading=current_heading,
            body=body,
            level=current_level,
        ))

    if not sections:
        sections = [Section(heading='', body=text.strip())]

    raw = '\n\n'.join(s.to_text() for s in sections)
    return DocumentStructure(
        sections=sections,
        title=title,
        format='md',
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# Plain text reader
# ---------------------------------------------------------------------------

def _read_plain(path: str) -> Optional[DocumentStructure]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    if not text.strip():
        return None

    return DocumentStructure(
        sections=[Section(heading='', body=text.strip())],
        format='txt',
        raw_text=text.strip(),
    )
