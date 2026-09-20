"""StructCT: local inference for structured clinical-trial matching."""

from .inference.pipeline import InferencePipeline, rerank_score

__all__ = ["InferencePipeline", "rerank_score"]
__version__ = "1.0.0"
