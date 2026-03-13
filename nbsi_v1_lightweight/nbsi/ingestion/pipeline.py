"""
nbsi/ingestion/pipeline.py

Multi-document ingestion pipeline.

This is the public API for Phase 4. It wires together:
    document_reader  → read with structure
    chunker          → split into meaningful sections
    metadata_nodes   → inject title/heading anchors
    spacy_extractor  → extract concepts per chunk
    session          → ingest all nodes + edges into unified graph

After ingest_documents(), the session graph contains concepts from every
document. Beam search will find paths that cross document boundaries.

Usage
-----
    from nbsi.ingestion.pipeline import ingest_documents, ingest_folder

    # Ingest a list of files
    report = ingest_documents(session, extractor, embedder,
                              paths=['paper.pdf', 'notes.md', 'report.docx'])

    # Ingest every supported file in a folder
    report = ingest_folder(session, extractor, embedder, folder='~/Documents')

    print(report.summary())

Both functions are synchronous. For background ingestion use the
IngestionWorker from os_integration/ingestion_worker.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class DocumentResult:
    """Result of ingesting a single document."""
    path:          str
    success:       bool
    format:        str       = ''
    chunks:        int       = 0
    nodes_added:   int       = 0
    edges_added:   int       = 0
    elapsed_s:     float     = 0.0
    error:         str       = ''

    def __str__(self):
        if not self.success:
            return f'  FAILED  {self.path}: {self.error}'
        return (f'  OK  {os.path.basename(self.path)}'
                f'  [{self.format}]'
                f'  {self.chunks} chunks'
                f'  {self.nodes_added} nodes'
                f'  {self.edges_added} edges'
                f'  ({self.elapsed_s:.1f}s)')


@dataclass
class IngestReport:
    """Aggregate result of ingesting multiple documents."""
    results:        list[DocumentResult] = field(default_factory=list)
    total_nodes:    int   = 0
    total_edges:    int   = 0
    files_ok:       int   = 0
    files_failed:   int   = 0
    elapsed_s:      float = 0.0

    def summary(self) -> str:
        lines = [
            f'Ingestion complete: {self.files_ok} ok, '
            f'{self.files_failed} failed, '
            f'{self.total_nodes} nodes, '
            f'{self.total_edges} edges, '
            f'{self.elapsed_s:.1f}s total',
        ]
        for r in self.results:
            lines.append(str(r))
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_documents(
    session,
    extractor,
    paths: list[str],
    verbose: bool = True,
) -> IngestReport:
    """
    Ingest a list of documents into the session graph.

    Parameters
    ----------
    session   : NBSISession
    extractor : SpacyExtractor
    paths     : list of file paths
    verbose   : print progress per document

    Returns IngestReport.
    """
    from nbsi.ingestion.document_reader import read_document
    from nbsi.ingestion.chunker         import chunk_document
    from nbsi.ingestion.metadata_nodes  import build_metadata_nodes, edges_from_chunk_to_heading

    report   = IngestReport()
    t_start  = time.monotonic()

    for path in paths:
        path = os.path.expanduser(path)
        t0   = time.monotonic()

        if verbose:
            print(f'[pipeline] Reading {os.path.basename(path)}...')

        # 1. Read with structure
        doc = read_document(path)
        if doc is None:
            r = DocumentResult(
                path=path, success=False,
                error='Could not read file (unsupported format or error)',
            )
            report.results.append(r)
            report.files_failed += 1
            continue

        # 2. Build metadata nodes (title, headings) — before chunking so we
        #    have the heading index ready to wire edges
        meta = build_metadata_nodes(doc, session.embedder)
        all_nodes = list(meta.nodes)
        all_edges = []

        # 3. Chunk and extract per section
        chunks_processed = 0
        for section in doc.sections:
            if section.is_empty():
                continue

            # Split the section body into chunks if it is large
            section_text = section.to_text()
            chunks = chunk_document(section_text)

            for chunk in chunks:
                if len(chunk.text.strip()) < 50:
                    continue

                # Extract content nodes for this chunk
                content_nodes, content_edges = extractor.extract(
                    chunk.text, session.embedder
                )

                if not content_nodes:
                    continue

                all_nodes.extend(content_nodes)
                all_edges.extend(content_edges)

                # Wire content nodes back to their section heading node
                heading_key = section.heading.strip().lower()
                heading_node = meta.heading_index.get(heading_key)
                if heading_node:
                    anchor_edges = edges_from_chunk_to_heading(
                        content_nodes, heading_node
                    )
                    all_edges.extend(anchor_edges)

                chunks_processed += 1

        if not all_nodes:
            r = DocumentResult(
                path=path, success=False,
                error='No nodes extracted from document',
            )
            report.results.append(r)
            report.files_failed += 1
            continue

        # 4. Ingest unified node+edge set into session
        result    = session.ingest_graph(all_nodes, all_edges)
        elapsed   = time.monotonic() - t0

        r = DocumentResult(
            path=path,
            success=True,
            format=doc.format,
            chunks=chunks_processed,
            nodes_added=result.get('nodes', len(all_nodes)),
            edges_added=result.get('edges', len(all_edges)),
            elapsed_s=elapsed,
        )
        report.results.append(r)
        report.files_ok    += 1
        report.total_nodes += r.nodes_added
        report.total_edges += r.edges_added

        if verbose:
            print(str(r))

    report.elapsed_s = time.monotonic() - t_start
    return report


def ingest_folder(
    session,
    extractor,
    folder: str,
    recursive: bool = True,
    verbose: bool = True,
) -> IngestReport:
    """
    Ingest every supported document in a folder.

    Parameters
    ----------
    session   : NBSISession
    extractor : SpacyExtractor
    folder    : path to the folder
    recursive : recurse into subdirectories
    verbose   : print progress per document
    """
    from nbsi.ingestion.document_reader import read_document

    SUPPORTED = {'.pdf', '.docx', '.html', '.htm', '.md', '.txt',
                 '.py', '.js', '.ts', '.rst'}

    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise ValueError(f'[pipeline] Not a directory: {folder}')

    paths = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in SUPPORTED:
                    paths.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in SUPPORTED:
                paths.append(fpath)

    if verbose:
        print(f'[pipeline] Found {len(paths)} document(s) in {folder}')

    return ingest_documents(session, extractor, paths, verbose=verbose)
