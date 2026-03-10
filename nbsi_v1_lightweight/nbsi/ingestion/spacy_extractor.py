"""
NBSI v1.0 Lightweight — spaCy Extractor

Reads plain text and produces ConceptNodes + ConceptEdges.
No API calls. No GPU. Runs entirely on CPU.

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""
import spacy
from nbsi.graph.node import ConceptNode
from nbsi.graph.edge import ConceptEdge

CAUSAL_VERBS   = {'cause','lead','produce','drive','trigger','generate','result','create','enable','prevent'}
CAUSAL_MARKS   = {'because','since','therefore','thus','hence','consequently','so that','as a result'}
CONTRAST_MARKS = {'however','but','although','despite','yet','whereas','nevertheless','on the other hand'}
CONSTIT_MARKS  = {'such as','including','for example','namely','consists of','made of','part of'}
SEQUENCE_MARKS = {'before','after','then','subsequently','first','next','finally','following','previously'}


class SpacyExtractor:
    """
    CPU-only extraction pipeline.
    Produces ConceptNodes and ConceptEdges from plain text.
    Embeddings are set by the embedder passed into extract().
    """
    def __init__(self, model: str = 'en_core_web_sm'):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found.\n"
                f"Install it with:  python -m spacy download {model}"
            )

    def extract(self, text: str, embedder) -> tuple[list, list]:
        """
        Extract nodes and edges from text.
        Returns (list[ConceptNode], list[ConceptEdge]).
        All nodes have real embeddings from the provided embedder.
        """
        doc = self.nlp(text)
        nodes: list[ConceptNode] = []
        edges: list[ConceptEdge] = []
        label_to_node: dict[str, ConceptNode] = {}

        # ── Collect candidate labels ──────────────────────────────────────
        labels = []
        chunk_labels = []
        for chunk in doc.noun_chunks:
            label = chunk.text.lower().strip()
            if len(label) < 3 or label in {
                'it','they','this','that','we','you','i','he','she',
                'its','their','our','your','his','her','which','who'
            }:
                continue
            if label not in label_to_node:
                labels.append(label)
                chunk_labels.append((label, chunk.root.ent_type_ != ''))

        # Named entities not already covered
        ent_labels = []
        for ent in doc.ents:
            label = ent.text.lower().strip()
            if label not in label_to_node and label not in labels:
                labels.append(label)
                ent_labels.append(label)

        if not labels:
            return [], []

        # ── Batch encode all labels in one call ───────────────────────────
        import numpy as np
        embeddings = embedder.encode(labels)

        # ── Build ConceptNodes ────────────────────────────────────────────
        for i, (label, emb) in enumerate(zip(labels, embeddings)):
            is_entity = (i < len(chunk_labels) and chunk_labels[i][1]) or label in ent_labels
            node = ConceptNode(
                label=label,
                embedding=emb.tolist(),
                node_type='entity' if is_entity else 'concept',
                activation=0.75 if is_entity else 0.70,
            )
            nodes.append(node)
            label_to_node[label] = node

        # ── Build ConceptEdges from dependency parse ──────────────────────
        for sent in doc.sents:
            edge_type = self._infer_edge_type(sent.text.lower(), sent)
            sent_chunks = [
                c for c in doc.noun_chunks
                if c.start >= sent.start and c.end <= sent.end
            ]
            for i in range(len(sent_chunks) - 1):
                src_label = sent_chunks[i].text.lower().strip()
                tgt_label = sent_chunks[i + 1].text.lower().strip()
                src = label_to_node.get(src_label)
                tgt = label_to_node.get(tgt_label)
                if src and tgt and src.node_id != tgt.node_id:
                    edges.append(ConceptEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        edge_type=edge_type,
                    ))

        return nodes, edges

    def _infer_edge_type(self, sent_text: str, sent) -> str:
        for m in CAUSAL_MARKS:
            if m in sent_text: return 'causal'
        for m in CONTRAST_MARKS:
            if m in sent_text: return 'contrastive'
        for m in CONSTIT_MARKS:
            if m in sent_text: return 'constitutive'
        for m in SEQUENCE_MARKS:
            if m in sent_text: return 'sequential'
        for tok in sent:
            if tok.lemma_ in CAUSAL_VERBS and tok.pos_ == 'VERB':
                return 'causal'
        return 'semantic'
