"""
nbsi/ingestion/chunker.py

Splits documents into meaningful sections before extraction.

The problem this solves:
    Feeding a long document to spaCy as one string produces too many
    nodes with weak cross-document connections. A 20,000 char white paper
    produces 838 nodes — mostly noise. An 80,000 char book chapter would
    produce thousands. Quality degrades as size grows.

The solution:
    Split first, extract per-chunk, then ingest all chunks into the same
    session graph. Each chunk produces a focused subgraph. The graph
    merges them via label-similarity deduplication (already in session.py).

Chunking strategy (in priority order):
    1. Explicit headings — markdown # / ## / ###, ALL CAPS lines, numbered
       sections like "1." or "2.1" — these are the best boundaries
    2. Blank-line paragraph groups — collect paragraphs until the group
       reaches TARGET_CHARS, then seal the chunk
    3. Sentence-boundary fallback — for dense prose with no paragraph
       breaks, split at sentence boundaries near TARGET_CHARS

Output:
    list of Chunk objects, each with:
        text      : str   — the body text for spaCy extraction
        heading   : str   — the section heading (or '' if none)
        position  : int   — ordinal index within document (0-based)
        char_start: int   — character offset in original text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_CHARS  = 3_000   # aim for chunks of roughly this size
MIN_CHARS     = 50      # chunks shorter than this are considered empty
MAX_CHARS     = 6_000   # hard ceiling — split even mid-paragraph if needed

# Heading patterns (applied in order — first match wins)
_HEADING_RE = re.compile(
    r'^(?:'
    r'(?P<md>#{1,4})\s+(?P<md_text>.+)'           # Markdown: # ## ### ####
    r'|(?P<num>(?:\d+\.)+\d*)\s+(?P<num_text>.+)' # Numbered: 1. / 2.1 / 3.2.1
    r'|(?P<cap>[A-Z][A-Z0-9 \-:]{8,})'            # ALL CAPS heading (min 9 chars)
    r')$',
    re.MULTILINE
)

# Sentence boundary — period / ! / ? followed by whitespace + capital
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text:       str
    heading:    str   = ''
    position:   int   = 0
    char_start: int   = 0

    def __len__(self):
        return len(self.text)

    def is_empty(self) -> bool:
        return len(self.text.strip()) < MIN_CHARS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(text: str) -> list[Chunk]:
    """
    Split a document into meaningful chunks for extraction.

    Returns a list of Chunk objects ordered by position.
    Always returns at least one chunk (the whole document) even if no
    splits are found.
    """
    if not text or not text.strip():
        return []

    # Try heading-based split first — highest quality
    chunks = _split_by_headings(text)

    # If headings produced too few or too many chunks, fall through to
    # paragraph-based grouping
    # Fall back only if heading split found NO headings at all
    has_headings = any(c.heading for c in chunks)
    if not has_headings and _quality_poor(chunks):
        chunks = _split_by_paragraphs(text)

    # Final safety: split any chunk that exceeds MAX_CHARS
    chunks = _enforce_max(chunks)

    # Number and return
    for i, c in enumerate(chunks):
        c.position = i

    return [c for c in chunks if c.text.strip()]


# ---------------------------------------------------------------------------
# Strategy 1 — heading-based splitting
# ---------------------------------------------------------------------------

def _split_by_headings(text: str) -> list[Chunk]:
    """
    Find all heading lines and use them as section boundaries.
    The text between two consecutive headings becomes one chunk.
    """
    lines   = text.splitlines(keepends=True)
    chunks  = []
    current_heading  = ''
    current_lines    = []
    current_start    = 0
    char_pos         = 0

    for line in lines:
        m = _HEADING_RE.match(line.rstrip())
        if m:
            # Seal the current section
            body = ''.join(current_lines).strip()
            if body:
                chunks.append(Chunk(
                    text=body,
                    heading=current_heading,
                    char_start=current_start,
                ))
            # Start new section
            current_heading = _extract_heading_text(m)
            current_lines   = []
            current_start   = char_pos
        else:
            current_lines.append(line)

        char_pos += len(line)

    # Seal final section
    body = ''.join(current_lines).strip()
    if body:
        chunks.append(Chunk(
            text=body,
            heading=current_heading,
            char_start=current_start,
        ))

    return chunks


def _extract_heading_text(m: re.Match) -> str:
    if m.group('md_text'):
        return m.group('md_text').strip()
    if m.group('num_text'):
        return m.group('num_text').strip()
    if m.group('cap'):
        return m.group('cap').strip()
    return ''


# ---------------------------------------------------------------------------
# Strategy 2 — paragraph-group splitting
# ---------------------------------------------------------------------------

def _split_by_paragraphs(text: str) -> list[Chunk]:
    """
    Group blank-line-separated paragraphs into chunks of ~TARGET_CHARS.
    """
    # Split on one or more blank lines
    paragraphs = re.split(r'\n{2,}', text)
    chunks     = []
    bucket     = []
    bucket_len = 0
    char_pos   = 0
    start      = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2  # blank lines
            continue

        bucket.append(para)
        bucket_len += len(para)
        char_pos   += len(para) + 2

        if bucket_len >= TARGET_CHARS:
            chunks.append(Chunk(
                text='\n\n'.join(bucket),
                char_start=start,
            ))
            start      = char_pos
            bucket     = []
            bucket_len = 0

    if bucket:
        chunks.append(Chunk(
            text='\n\n'.join(bucket),
            char_start=start,
        ))

    return chunks if chunks else [Chunk(text=text.strip())]


# ---------------------------------------------------------------------------
# Safety: enforce MAX_CHARS
# ---------------------------------------------------------------------------

def _enforce_max(chunks: list[Chunk]) -> list[Chunk]:
    """Split any chunk that exceeds MAX_CHARS at a sentence boundary."""
    result = []
    for chunk in chunks:
        if len(chunk.text) <= MAX_CHARS:
            result.append(chunk)
            continue
        # Split at sentence boundaries
        result.extend(_split_long_chunk(chunk))
    return result


def _split_long_chunk(chunk: Chunk) -> list[Chunk]:
    """Split a single oversized chunk at sentence boundaries."""
    text     = chunk.text
    parts    = []
    start    = 0

    while start < len(text):
        end = start + MAX_CHARS
        if end >= len(text):
            parts.append(text[start:].strip())
            break

        # Find the last sentence boundary before MAX_CHARS
        segment = text[start:end]
        matches = list(_SENT_BOUNDARY.finditer(segment))
        if matches:
            cut   = matches[-1].start() + 1
            parts.append(text[start : start + cut].strip())
            start = start + cut
        else:
            # No sentence boundary — cut at whitespace
            ws = segment.rfind(' ')
            cut = ws if ws > 0 else MAX_CHARS
            parts.append(text[start : start + cut].strip())
            start = start + cut

    return [
        Chunk(
            text=p,
            heading=chunk.heading,
            char_start=chunk.char_start + i * MAX_CHARS,  # approximate
        )
        for i, p in enumerate(parts)
        if p
    ]


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def _quality_poor(chunks: list[Chunk]) -> bool:
    """
    True if the heading split produced results too skewed to be useful:
    - One chunk contains more than 80% of the total text, OR
    - Average chunk is still over MAX_CHARS
    """
    if not chunks:
        return True
    total = sum(len(c.text) for c in chunks)
    if total == 0:
        return True
    largest   = max(len(c.text) for c in chunks)
    avg       = total / len(chunks)
    return (largest / total > 0.8) or (avg > MAX_CHARS)
