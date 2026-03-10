"""
nbsi/ingestion/metadata_nodes.py

Injects document metadata as protected anchor nodes before extraction.

The problem without metadata nodes:
    The graph contains only content concepts. There is no way to query
    "what did the introduction say" or "which document mentioned X" because
    those structural facts are not in the graph. Every concept floats free
    of its document origin.

What metadata nodes do:
    Title, author, section headings become ConceptNodes with:
        - node_type = 'metadata'
        - activation = 0.9  (higher than content nodes at 0.7)
        - protected = True  (not pruned by MAX_NODES cull)

    The extractor then creates edges from content nodes back to the section
    heading node they came from. This anchors every concept to its source
    section. Beam search can now find "introduction → concept → evidence"
    paths.

Usage:
    from nbsi.ingestion.metadata_nodes import build_metadata_nodes

    meta_nodes = build_metadata_nodes(doc_structure, embedder)
    # meta_nodes is a list of ConceptNode
    # pass them to session.ingest_graph() along with content nodes

The section heading nodes are also returned in a lookup dict so the
extractor can wire edges to them:

    meta_nodes, heading_index = build_metadata_nodes(doc_structure, embedder)
    # heading_index: {section_heading_str: ConceptNode}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MetadataNodeSet:
    """Container for metadata nodes and the heading lookup."""
    nodes:         list   # list[ConceptNode]
    heading_index: dict   # {heading_text: ConceptNode}
    title_node:    object = None  # ConceptNode or None
    author_node:   object = None  # ConceptNode or None


def build_metadata_nodes(doc_structure, embedder) -> MetadataNodeSet:
    """
    Build metadata ConceptNodes from a DocumentStructure.

    Parameters
    ----------
    doc_structure : DocumentStructure  (from document_reader.py)
    embedder      : any embedder with .encode(texts) -> np.ndarray

    Returns
    -------
    MetadataNodeSet
    """
    # Import here to keep this module usable in tests without the full stack
    from nbsi.graph.node import ConceptNode

    nodes         = []
    heading_index = {}
    title_node    = None
    author_node   = None

    labels_to_embed = []
    label_roles     = []   # 'title' | 'author' | 'heading'

    # Collect labels for batch embedding
    if doc_structure.title and doc_structure.title.strip():
        labels_to_embed.append(doc_structure.title.strip().lower())
        label_roles.append('title')

    if doc_structure.author and doc_structure.author.strip():
        labels_to_embed.append(doc_structure.author.strip().lower())
        label_roles.append('author')

    seen_headings = set()
    for section in doc_structure.sections:
        h = section.heading.strip()
        if h and h.lower() not in seen_headings:
            labels_to_embed.append(h.lower())
            label_roles.append('heading')
            seen_headings.add(h.lower())

    if not labels_to_embed:
        return MetadataNodeSet(nodes=[], heading_index={})

    # Batch embed all metadata labels
    embeddings = embedder.encode(labels_to_embed)

    for label, role, emb in zip(labels_to_embed, label_roles, embeddings):
        node = ConceptNode(
            label=label,
            embedding=emb.tolist(),
            node_type='metadata',
            activation=0.9,
        )
        # Mark as protected so the pruner never removes it
        node.protected = True

        nodes.append(node)

        if role == 'title':
            title_node = node
        elif role == 'author':
            author_node = node
        elif role == 'heading':
            heading_index[label] = node

    return MetadataNodeSet(
        nodes=nodes,
        heading_index=heading_index,
        title_node=title_node,
        author_node=author_node,
    )


def edges_from_chunk_to_heading(chunk_nodes: list,
                                heading_node,
                                edge_weight: float = 0.8) -> list:
    """
    Create ConceptEdges from each content node in a chunk to the
    section heading node that chunk belongs to.

    This anchors every concept to its source section so beam search
    can find section-level paths.

    Parameters
    ----------
    chunk_nodes   : content nodes extracted from one chunk
    heading_node  : the metadata ConceptNode for that chunk's heading
    edge_weight   : conductivity weight for the anchor edges (default 0.8)

    Returns list of ConceptEdge
    """
    if heading_node is None:
        return []

    from nbsi.graph.edge import ConceptEdge

    edges = []
    for node in chunk_nodes:
        if node.node_id == heading_node.node_id:
            continue
        edges.append(ConceptEdge(
            source_id=node.node_id,
            target_id=heading_node.node_id,
            edge_type='constitutive',
        ))
    return edges
