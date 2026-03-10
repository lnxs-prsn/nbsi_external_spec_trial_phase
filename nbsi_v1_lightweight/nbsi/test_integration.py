from nbsi.embedder import RealEmbedder
from nbsi.ingestion.spacy_extractor import SpacyExtractor
from nbsi.ingestion.pipeline import ingest_documents
from nbsi.session.session import NBSISession
from nbsi.lifecycle.structural_library import StructuralNodeLibrary
from nbsi.config import Config

config = Config()
config.MAX_NODES = 700
session   = NBSISession(StructuralNodeLibrary(), RealEmbedder(), config)
extractor = SpacyExtractor()

report = ingest_documents(session, extractor, ['NBSI_External_Paper.docx'])
print(report.summary())
print('Graph nodes:', session.graph.node_count)
