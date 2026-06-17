"""ambit — visualize how a dataset occupies an embedding space.

ambit is a library first; the `ambit` command is one front end over it. The common
entry points are the three high-level verbs, which return values (not terminal text)
so they compose anywhere — a notebook, a service, a batch job, or the CLI:

    import ambit
    rep  = ambit.report("embeddings.parquet")     # -> Report (.html, .ctx, .write(path))
    diag = ambit.diagnose("embeddings.parquet")   # -> Diagnostics (.mean_cos, .verdict, ...)
    ambit.embed("items.jsonl", "vecs.parquet", model="text-embedding-3-small")

Each takes a `Config` or plain keyword overrides. The building blocks underneath
(`scan`, `build_ctx`, `build_report`, `metrics`, `localized_anisotropy`, ...) are
exported too, for callers that want to drive the pipeline stage by stage.
"""

from . import metrics
from .config import Config
from .embedding import EmbeddingClient, embed_dataset
from .ingest import load
from .local_anisotropy import LocalAnisotropy, localized_anisotropy
from .pipeline import Ctx, build_ctx
from .render import build_report
from .scan import Scan, scan
from .types import EmbeddingSet

# the high-level verbs (the embedding submodule is `ambit.embedding`, so the `embed`
# verb here is unambiguous and does not shadow a module)
from .api import (  # noqa: E402
    Diagnostics,
    Report,
    build_context,
    diagnose,
    embed,
    report,
)

__version__ = "0.0.1"
__all__ = [
    # high-level verbs
    "report", "diagnose", "embed", "build_context",
    "Report", "Diagnostics",
    # run configuration
    "Config",
    # pipeline building blocks
    "load", "scan", "Scan", "build_ctx", "Ctx", "build_report",
    "localized_anisotropy", "LocalAnisotropy", "metrics",
    # embedding
    "EmbeddingClient", "embed_dataset",
    # core type
    "EmbeddingSet",
    "__version__",
]
