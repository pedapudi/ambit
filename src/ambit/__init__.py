"""ambit — visualize how a dataset occupies an embedding space."""

from .embed import EmbeddingClient, embed_dataset
from .ingest import load
from .scan import Scan, scan
from .types import EmbeddingSet

__version__ = "0.0.1"
__all__ = ["EmbeddingSet", "load", "scan", "Scan", "EmbeddingClient", "embed_dataset", "__version__"]
