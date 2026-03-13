"""
nbsi/os_integration/reader.py

Reads any supported file format to plain text.
Returns None on failure — never raises.

Supported formats:
    .txt  .md  .py  .js  .ts  .rst   — stdlib open()
    .pdf                              — pdfminer.six
    .docx                             — python-docx
    .ics                              — icalendar
"""

import os


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------

def read_file(path: str) -> str | None:
    """
    Read any supported file to plain text.
    Returns None if the file cannot be read or is unsupported.
    """
    if not os.path.isfile(path):
        return None

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in ('.txt', '.md', '.py', '.js', '.ts', '.rst', '.csv'):
            return _read_text(path)
        elif ext == '.pdf':
            return _read_pdf(path)
        elif ext == '.docx':
            return _read_docx(path)
        elif ext == '.ics':
            return _read_ics(path)
        else:
            return None  # unsupported — not an error
    except Exception as e:
        print(f'[reader] Failed to read {path}: {e}')
        return None


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _read_pdf(path: str) -> str | None:
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print('[reader] pdfminer.six not installed — cannot read PDF. '
              'Install with: pip install pdfminer.six')
        return None

    text = extract_text(path)
    return text if text and text.strip() else None


def _read_docx(path: str) -> str | None:
    try:
        from docx import Document
    except ImportError:
        print('[reader] python-docx not installed — cannot read DOCX. '
              'Install with: pip install python-docx')
        return None

    doc = Document(path)

    sections = []

    # Headings preserved with a blank line before them so spaCy sees
    # paragraph boundaries rather than one massive run-on block.
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ''
        if 'Heading' in style:
            sections.append('\n' + text)
        else:
            sections.append(text)

    return '\n'.join(sections) if sections else None


def _read_ics(path: str) -> str | None:
    try:
        from icalendar import Calendar
    except ImportError:
        print('[reader] icalendar not installed — cannot read ICS. '
              'Install with: pip install icalendar')
        return None

    with open(path, 'rb') as f:
        cal = Calendar.from_ical(f.read())

    lines = []
    for component in cal.walk():
        if component.name == 'VEVENT':
            dtstart = component.get('DTSTART', '')
            summary = component.get('SUMMARY', '')
            description = component.get('DESCRIPTION', '')
            # Emit as a short prose sentence — natural input for spaCy
            parts = [str(p) for p in [dtstart, summary, description] if p]
            lines.append('. '.join(parts))

    return '\n'.join(lines) if lines else None


# ---------------------------------------------------------------------------
# Batch helper — used by IngestionWorker
# ---------------------------------------------------------------------------

def read_files(paths: list[str]) -> dict[str, str]:
    """
    Read multiple files. Returns {path: text} for every file that
    succeeded. Silently drops files that returned None.
    """
    results = {}
    for path in paths:
        text = read_file(path)
        if text and len(text.strip()) >= 50:
            results[path] = text
    return results


# ---------------------------------------------------------------------------
# Supported extension set — used by watcher to filter events
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.py', '.js', '.ts', '.rst', '.csv',
    '.pdf', '.docx', '.ics',
}
